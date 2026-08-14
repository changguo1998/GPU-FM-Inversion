# forward_test.jl — Stage 3 (forward C++ kernel) tests.
#
# Runs `forward` on a small hand-built trial set (36–72, NOT the full 151,848
# grid) and checks `cc_max`/`best_lag` against an independent Julia reference
# recomputing the XCorr math from the stored obs/GF windows (same kernel
# definitions: window-internal zero-padded shifts, per-lag normalization) —
# validating MT layout and lag handling exactly. Also asserts `/intermediates`
# idempotence (a re-run over the same status file replaces cleanly).
#
# Usage:
#   julia --project=. tests/stages/forward_test.jl

using Test
using HDF5

using IO, Grid

include("test_util.jl")

# ── Independent Julia reference: replicate the C++ XCorr math ─────────────
# cc_norm[lag] = (mᵀ·dot_lag) / sqrt(obs_n2 · mᵀ·synamp_lag·m)
#   dot_lag[c] = Σ_t obs[t+lag]·gf[t,c]
#   synamp_lag = Σ_t gf[t−lag,a]·gf[t−lag,b] (GF shifted by −lag)
# zero-padded shifts; lag ∈ [−maxlag, +maxlag].

"""
    reference_cc(obs, gf_mat, m, obs_n2, maxlag) -> (cc_max, best_lag)

Recompute one (phase, trial) XCorr pair exactly as the C++ kernel does.
`gf_mat` is [6, N] (component × time); `obs` is [N].
"""
function reference_cc(
    obs::Vector{Float64},
    gf_mat::Matrix{Float64},
    m::Vector{Float64},
    obs_n2::Float64,
    maxlag::Int,
)
    N = length(obs)
    best = 0.0
    bestl = 0
    for lag in (-maxlag):maxlag
        # per-lag synamp (GF shifted by -lag) quadratic form
        syn = 0.0
        for a in 1:6, b in a:6
            s = 0.0
            for t in 0:(N - 1)
                ts = t - lag
                (0 <= ts < N) || continue
                s += gf_mat[a, ts + 1] * gf_mat[b, ts + 1]
            end
            syn += (a == b ? 1 : 2) * m[a] * m[b] * s
        end
        syn <= 0.0 && continue
        # per-lag dot (obs shifted by +lag, GF fixed)
        cc = 0.0
        for comp in 1:6
            s = 0.0
            for t in 0:(N - 1)
                ts = t + lag
                (0 <= ts < N) || continue
                s += obs[ts + 1] * gf_mat[comp, t + 1]
            end
            cc += m[comp] * s
        end
        cn = cc / sqrt(obs_n2 * syn)
        if abs(cn) > best
            best = abs(cn)
            bestl = lag
        end
    end
    return best, bestl
end

@testset "forward stage" begin
    mktempdir() do dir
        make_synthetic(dir; nsta = 3)
        config = write_test_config(dir)

        r = run_stage_script(joinpath("scripts", "input.jl"), [config])
        @test r.ok

        db = joinpath(dir, "database.h5")
        status_dir = joinpath(dir, "status")
        mkpath(status_dir)
        status0 = joinpath(status_dir, "status_0.h5")
        mv(joinpath(dir, "status_0.h5"), status0)

        # ── Small trial set: 3×2×2 SDR grid × 3 depths × 1 freq = 36 trials ──
        strat = IO.Strategy(
            0.0,
            5.0,
            Int32(3),
            0.0,
            5.0,
            Int32(2),
            -90.0,
            5.0,
            Int32(2),
            Int32[1, 2, 3],
            Int32[1],
            Int32(0),
        )
        trials = Grid.generate_trials(strat)
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)
        @test N_trials == 36

        # ── Run forward (twice: schema + idempotence) ──
        exe = joinpath(PROJECT_ROOT, "forward", "build", "forward")
        fwd = run_cmd(`$exe $db $status0`)
        @testset "run" begin
            @test fwd.ok
            @test !occursin("HDF5-DIAG", fwd.stderr)  # no H5Lexists noise
            @test isempty(fwd.stderr)                 # clean stderr
            @test isfile(status0)
        end
        fwd2 = run_cmd(`$exe $db $status0`)
        @test fwd2.ok  # idempotent rewrite

        # ── Read database intermediates for the reference ──
        h5open(db, "r") do f
            global _OBS = read(f["/XcorrS/obs/1/obs"])
            global _GFS = Dict(
                1 => read(f["/XcorrS/gf/1/1/gf"]),  # Julia layout [N, 6, samples]
                2 => read(f["/XcorrS/gf/2/1/gf"]),
                3 => read(f["/XcorrS/gf/3/1/gf"]),
            )
            global _OBSN2 = read(f["/XcorrS/obs/1/obs_norm2"])
            global _STRIKE = read(f["/paraspace/strike"])
            global _DIP = read(f["/paraspace/dip"])
            global _RAKE = read(f["/paraspace/rake"])
            global _NENT = length(_OBSN2)
        end

        h5open(status0, "r") do f
            global _CC = read(f["/intermediates/XcorrS/cc_max"])
            global _LAG = read(f["/intermediates/XcorrS/best_lag"])
            global _TS = read(f["/trials/strike_idx"])
            global _TD = read(f["/trials/dip_idx"])
            global _TR = read(f["/trials/rake_idx"])
            global _TDEP = read(f["/trials/depth_idx"])
        end

        @testset "output shapes" begin
            @test size(_CC) == (N_trials, _NENT)  # HDF5.jl reads C-order [N,phases]
            @test size(_LAG) == (N_trials, _NENT)
            @test all(0.0 .<= _CC .<= 1.0)
            @test all(-150 .<= _LAG .<= 150)
        end

        @testset "reference match (kernel math + MT layout)" begin
            max_dcc = 0.0
            max_dlag = 0
            for t in 1:N_trials
                dep = Int(_TDEP[t])
                gf3 = _GFS[dep]
                m = sdr_to_mt(_STRIKE[_TS[t]], _DIP[_TD[t]], _RAKE[_TR[t]])
                for ph in 1:_NENT
                    rb, rl = reference_cc(vec(_OBS[ph, :]), gf3[ph, :, :], m, _OBSN2[ph], 150)
                    fb = _CC[t, ph]
                    fl = _LAG[t, ph]
                    @test rb ≈ fb atol = 1e-9
                    @test rl == fl
                    max_dcc = max(max_dcc, abs(rb - fb))
                    max_dlag = max(max_dlag, abs(rl - fl))
                end
            end
            @test max_dcc < 1e-9
            @test max_dlag == 0
        end
    end
end


@testset "window-clamp regression (maxlag past half window)" begin
    mktempdir() do dir
        make_synthetic(dir; nsta = 3)
        config = write_test_config(dir)
        # Request max_lag_periods=6.0 -> configured maxlag = 6.0/2.0/0.01 = 300,
        # which exceeds (501-1)/2 = 250; DataCache must clamp to 250.
        cfg = read(config, String)
        cfg = replace(
            cfg,
            "Config.XcorrS.max_lag_periods() = 3.0" => "Config.XcorrS.max_lag_periods() = 6.0",
        )
        write(config, cfg)

        r = run_stage_script(joinpath("scripts", "input.jl"), [config])
        @test r.ok
        db = joinpath(dir, "database.h5")
        status_dir = joinpath(dir, "status")
        mkpath(status_dir)
        status0 = joinpath(status_dir, "status_0.h5")
        mv(joinpath(dir, "status_0.h5"), status0)

        strat = IO.Strategy(
            0.0,
            5.0,
            Int32(3),
            0.0,
            5.0,
            Int32(2),
            -90.0,
            5.0,
            Int32(2),
            Int32[1, 2, 3],
            Int32[1],
            Int32(0),
        )
        trials = Grid.generate_trials(strat)
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)

        exe = joinpath(PROJECT_ROOT, "forward", "build", "forward")
        fwd = run_cmd(`$exe $db $status0`)
        @test fwd.ok  # must not crash / overrun the clamped allocation

        h5open(db, "r") do f
            global _GFS_CL = Dict(
                1 => read(f["/XcorrS/gf/1/1/gf"]),
                2 => read(f["/XcorrS/gf/2/1/gf"]),
                3 => read(f["/XcorrS/gf/3/1/gf"]),
            )
            global _OBS_CL = read(f["/XcorrS/obs/1/obs"])
            global _OBSN2_CL = read(f["/XcorrS/obs/1/obs_norm2"])
            global _STRIKE_CL = read(f["/paraspace/strike"])
            global _DIP_CL = read(f["/paraspace/dip"])
            global _RAKE_CL = read(f["/paraspace/rake"])
        end
        h5open(status0, "r") do f
            global _CC_CL = read(f["/intermediates/XcorrS/cc_max"])
            global _LAG_CL = read(f["/intermediates/XcorrS/best_lag"])
            global _TS_CL = read(f["/trials/strike_idx"])
            global _TD_CL = read(f["/trials/dip_idx"])
            global _TR_CL = read(f["/trials/rake_idx"])
            global _DEP_CL = read(f["/trials/depth_idx"])
        end

        n_ent = size(_CC_CL, 2)
        @test n_ent >= 1
        # lag range clamped to ±(501-1)/2 = ±250, not the requested 300
        @test extrema(_LAG_CL) ⊆ (-250:250)
        @test all(0.0 .<= _CC_CL .<= 1.0)

        max_dcc = 0.0
        for t in 1:N_trials
            dep = Int(_DEP_CL[t])
            gf3 = _GFS_CL[dep]
            m = sdr_to_mt(_STRIKE_CL[_TS_CL[t]], _DIP_CL[_TD_CL[t]], _RAKE_CL[_TR_CL[t]])
            for ph in 1:n_ent
                rb, rl = reference_cc(vec(_OBS_CL[ph, :]), gf3[ph, :, :], m, _OBSN2_CL[ph], 250)
                @test rb ≈ _CC_CL[t, ph] atol = 1e-9
                @test rl == _LAG_CL[t, ph]
                max_dcc = max(max_dcc, abs(rb - _CC_CL[t, ph]))
            end
        end
        @test max_dcc < 1e-9
    end
end
