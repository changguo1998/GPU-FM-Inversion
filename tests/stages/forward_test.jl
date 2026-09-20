# forward_test.jl — Stage 3 (forward C++ kernel) tests.
#
# Runs `forward` on a small hand-built trial set (108–216, NOT the full 455,544
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

using IO, Search

include("test_util.jl")

@testset "forward CLI validation" begin
    exe = forward_executable()

    missing = run_cmd(Cmd([exe]))
    @test !missing.ok
    @test occursin("Usage: forward", missing.stderr)

    unknown = run_cmd(Cmd([exe, "--unknown", "database.h5", "status.h5"]))
    @test !unknown.ok
    @test occursin("unknown option", unknown.stderr)

    invalid_backend = run_cmd(Cmd([exe, "--backend", "invalid", "database.h5", "status.h5"]))
    @test !invalid_backend.ok
    @test occursin("invalid backend", invalid_backend.stderr)

    for batch in ("0", "-1", "abc")
        invalid_batch = run_cmd(
            Cmd([
                exe,
                "--backend",
                "cuda",
                "--cuda-batch-trials",
                batch,
                "database.h5",
                "status.h5",
            ]),
        )
        @test !invalid_batch.ok
        @test occursin("positive integer", invalid_batch.stderr)
    end

    if get(ENV, "FM_TEST_CUDA", "0") != "1"
        cuda = run_cmd(Cmd([exe, "--backend", "cuda", "database.h5", "status.h5"]))
        @test !cuda.ok
        @test occursin("CUDA backend not compiled", cuda.stderr)
    else
        hidden_auto = run_cmd(
            addenv(
                Cmd([exe, "--backend", "auto", "database.h5", "status.h5"]),
                "CUDA_VISIBLE_DEVICES" => "",
            ),
        )
        @test !hidden_auto.ok
        @test occursin("selected backend=cpu", hidden_auto.stdout)

        hidden_cuda = run_cmd(
            addenv(
                Cmd([exe, "--backend", "cuda", "database.h5", "status.h5"]),
                "CUDA_VISIBLE_DEVICES" => "",
            ),
        )
        @test !hidden_cuda.ok
        @test occursin("no CUDA device available", hidden_cuda.stderr)

        probe_error = run_cmd(
            addenv(
                Cmd([exe, "--backend", "auto", "database.h5", "status.h5"]),
                "FM_CUDA_TEST_PROBE_ERROR" => "1",
            ),
        )
        @test !probe_error.ok
        @test occursin("injected CUDA initialization failure", probe_error.stderr)
        @test !occursin("selected backend=cpu", probe_error.stdout)
    end

    cpu_batch = run_cmd(
        Cmd([exe, "--backend", "cpu", "--cuda-batch-trials", "1", "database.h5", "status.h5"]),
    )
    @test !cpu_batch.ok
    @test occursin("requires CUDA backend", cpu_batch.stderr)
end

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
    found = false
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
        if !found || cn > best
            best = cn
            bestl = lag
            found = true
        end
    end
    return best, bestl
end

function replace_dataset!(path::AbstractString, dataset::AbstractString, value)
    h5open(path, "r+") do f
        delete_object(f, dataset)
        write(f, dataset, value)
    end
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

        # ── Small trial set: 3×2×2 SDR × 3 depths × 1 freq × 3 durations = 108 ──
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
            Int32[1, 2, 3],
            Int32(0),
        )
        generated_trials = Search.generate_trials(strat)
        order =
            vcat(2:2:length(generated_trials.strike_idx), 1:2:length(generated_trials.strike_idx))
        trials = IO.TrialSet(
            generated_trials.strike_idx[order],
            generated_trials.dip_idx[order],
            generated_trials.rake_idx[order],
            generated_trials.depth_idx[order],
            generated_trials.freq_idx[order],
            generated_trials.duration_idx[order],
        )
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)
        @test N_trials == 108
        status_template = joinpath(status_dir, "status_template.h5")
        cp(status0, status_template)

        # ── Run forward (twice: schema + idempotence) ──
        test_cuda = get(ENV, "FM_TEST_CUDA", "0") == "1"
        fwd = run_forward(db, status0; backend = test_cuda ? "cpu" : nothing)
        @testset "run" begin
            @test fwd.ok
            expected_backend = test_cuda ? "requested backend=cpu" : "requested backend=auto"
            @test occursin("$expected_backend, selected backend=cpu", fwd.stdout)
            @test !occursin("HDF5-DIAG", fwd.stderr)  # no H5Lexists noise
            @test isempty(fwd.stderr)                 # clean stderr
            @test isfile(status0)
        end
        fwd2 = run_forward(db, status0; backend = "cpu", env = Dict("FM_FORWARD_BACKEND" => "cuda"))
        @test fwd2.ok  # idempotent rewrite
        @test occursin("requested backend=cpu, selected backend=cpu", fwd2.stdout)
        h5open(status0, "r") do f
            @test !haskey(f, "/intermediates.__tmp__")
            @test !haskey(f, "/intermediates.__backup__")
        end

        # ── Read database intermediates for the reference ──
        h5open(db, "r") do f
            global _XCORR_INPUTS = Dict(
                key => (
                    obs = read(f["/$key/obs/1/obs"]),
                    gfs = Dict(
                        (dep, duration) => read(f["/$key/gf/$dep/1/$duration/gf"]) for
                        dep in 1:3 for duration in 1:3
                    ),
                    obs_norm2 = read(f["/$key/obs/1/obs_norm2"]),
                ) for key in ("XcorrP", "XcorrS")
            )
            global _STRIKE = read(f["/paraspace/strike"])
            global _DIP = read(f["/paraspace/dip"])
            global _RAKE = read(f["/paraspace/rake"])
            global _NENT = length(_XCORR_INPUTS["XcorrS"].obs_norm2)
        end

        h5open(status0, "r") do f
            global _CC = read(f["/intermediates/XcorrS/cc_max"])
            global _LAG = read(f["/intermediates/XcorrS/best_lag"])
            global _TS = read(f["/trials/strike_idx"])
            global _TD = read(f["/trials/dip_idx"])
            global _TR = read(f["/trials/rake_idx"])
            global _TDEP = read(f["/trials/depth_idx"])
            global _TDURATION = read(f["/trials/duration_idx"])
        end

        @testset "output shapes" begin
            @test size(_CC) == (N_trials, _NENT)  # HDF5.jl reads C-order [N,phases]
            @test size(_LAG) == (N_trials, _NENT)
            @test all(-1.0 .<= _CC .<= 1.0)
            @test all(-150 .<= _LAG .<= 150)
        end

        @testset "reference match (kernel math + MT layout)" begin
            max_dcc = 0.0
            max_dlag = 0
            xcorr_s = _XCORR_INPUTS["XcorrS"]
            for t in 1:N_trials
                dep = Int(_TDEP[t])
                gf3 = xcorr_s.gfs[(dep, Int(_TDURATION[t]))]
                m = sdr_to_mt(_STRIKE[_TS[t]], _DIP[_TD[t]], _RAKE[_TR[t]])
                for ph in 1:_NENT
                    rb, rl = reference_cc(
                        vec(xcorr_s.obs[ph, :]),
                        gf3[ph, :, :],
                        m,
                        xcorr_s.obs_norm2[ph],
                        150,
                    )
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

        if test_cuda
            @testset "CUDA parity" begin
                cpu_outputs = h5open(status0, "r") do f
                    Dict(
                        (key, dataset) => read(f["/intermediates/$key/$dataset"]) for
                        key in ("XcorrP", "XcorrS") for dataset in ("cc_max", "best_lag")
                    )
                end
                for batch in (nothing, 1, 7, 8)
                    cuda_status = joinpath(dir, "status_cuda_$(something(batch, "auto")).h5")
                    cp(status_template, cuda_status)
                    cuda_run = run_forward(db, cuda_status; backend = "cuda", batch_trials = batch)
                    @test cuda_run.ok
                    @test occursin("selected backend=cuda", cuda_run.stdout)
                    h5open(cuda_status, "r") do f
                        for key in ("XcorrP", "XcorrS")
                            cc = read(f["/intermediates/$key/cc_max"])
                            lag = read(f["/intermediates/$key/best_lag"])
                            @test eltype(cc) == Float64
                            @test eltype(lag) == Int32
                            @test size(cc) == size(cpu_outputs[(key, "cc_max")])
                            @test size(lag) == size(cpu_outputs[(key, "best_lag")])
                            @test maximum(abs.(cc .- cpu_outputs[(key, "cc_max")])) <= 1e-9
                            @test lag == cpu_outputs[(key, "best_lag")]

                            inputs = _XCORR_INPUTS[key]
                            for t in 1:N_trials
                                gf = inputs.gfs[(Int(_TDEP[t]), Int(_TDURATION[t]))]
                                m = sdr_to_mt(_STRIKE[_TS[t]], _DIP[_TD[t]], _RAKE[_TR[t]])
                                for ph in eachindex(inputs.obs_norm2)
                                    rb, rl = reference_cc(
                                        vec(inputs.obs[ph, :]),
                                        gf[ph, :, :],
                                        m,
                                        inputs.obs_norm2[ph],
                                        150,
                                    )
                                    @test cc[t, ph] ≈ rb atol = 1e-9
                                    @test lag[t, ph] == rl
                                end
                            end
                        end
                    end
                    if batch === nothing
                        repeated = run_forward(db, cuda_status; backend = "cuda")
                        @test repeated.ok
                        @test isempty(repeated.stderr)
                        h5open(cuda_status, "r") do f
                            @test !haskey(f, "/intermediates.__tmp__")
                            @test !haskey(f, "/intermediates.__backup__")
                            for key in ("XcorrP", "XcorrS"), dataset in ("cc_max", "best_lag")
                                @test read(f["/intermediates/$key/$dataset"]) ==
                                      cpu_outputs[(key, dataset)]
                            end
                        end
                    end
                end

                @testset "CUDA failure preserves old intermediates" begin
                    for failure_batch in (0, 1)
                        failed_status = joinpath(dir, "status_cuda_fail_$failure_batch.h5")
                        cp(status0, failed_status)
                        marked_cc = copy(cpu_outputs[("XcorrP", "cc_max")])
                        marked_cc[1] = 123.0 + failure_batch
                        replace_dataset!(failed_status, "/intermediates/XcorrP/cc_max", marked_cc)

                        failed = run_forward(
                            db,
                            failed_status;
                            backend = "cuda",
                            batch_trials = 1,
                            env = Dict("FM_CUDA_TEST_FAIL_BATCH" => string(failure_batch)),
                        )
                        @test !failed.ok
                        @test occursin("injected CUDA batch failure", failed.stderr)
                        h5open(failed_status, "r") do f
                            @test read(f["/intermediates/XcorrP/cc_max"])[1] ==
                                  123.0 + failure_batch
                            @test !haskey(f, "/intermediates.__tmp__")
                            @test !haskey(f, "/intermediates.__backup__")
                        end

                        recovered =
                            run_forward(db, failed_status; backend = "cuda", batch_trials = 1)
                        @test recovered.ok
                        h5open(failed_status, "r") do f
                            @test read(f["/intermediates/XcorrP/cc_max"]) ==
                                  cpu_outputs[("XcorrP", "cc_max")]
                        end
                    end
                end

                if get(ENV, "FM_TEST_SANITIZER", "0") == "1"
                    @testset "CUDA memcheck" begin
                        sanitized_status = joinpath(dir, "status_cuda_memcheck.h5")
                        cp(status_template, sanitized_status)
                        sanitizer = get(ENV, "FM_COMPUTE_SANITIZER", "compute-sanitizer")
                        sanitized = run_cmd(
                            Cmd([
                                sanitizer,
                                "--tool",
                                "memcheck",
                                forward_executable(),
                                "--backend",
                                "cuda",
                                "--cuda-batch-trials",
                                "7",
                                db,
                                sanitized_status,
                            ]),
                        )
                        @test sanitized.ok
                        sanitizer_output = sanitized.stdout * sanitized.stderr
                        @test occursin("ERROR SUMMARY: 0 errors", sanitizer_output)
                    end
                end
            end
        end

        @testset "preflight rejects invalid input" begin
            short_status = joinpath(dir, "status_short.h5")
            cp(status0, short_status)
            strike_idx = h5open(short_status, "r") do f
                read(f["/trials/strike_idx"])
            end
            replace_dataset!(short_status, "/trials/strike_idx", strike_idx[1:(end - 1)])
            short_run = run_forward(db, short_status)
            @test !short_run.ok
            @test occursin("trial index length mismatch for strike", short_run.stderr)

            bad_index_status = joinpath(dir, "status_bad_index.h5")
            cp(status0, bad_index_status)
            depth_idx = h5open(bad_index_status, "r") do f
                read(f["/trials/depth_idx"])
            end
            depth_idx[1] = 0
            replace_dataset!(bad_index_status, "/trials/depth_idx", depth_idx)
            bad_index_run = run_forward(db, bad_index_status)
            @test !bad_index_run.ok
            @test occursin("invalid depth index", bad_index_run.stderr)

            mismatch_db = joinpath(dir, "database_config_mismatch.h5")
            cp(db, mismatch_db)
            replace_dataset!(mismatch_db, "/config/XcorrP/max_lag_periods", 4.0)
            mismatch_run = run_forward(mismatch_db, status0)
            @test !mismatch_run.ok
            @test occursin("XCorr P/S configuration mismatch", mismatch_run.stderr)

            dt_mismatch_db = joinpath(dir, "database_dt_mismatch.h5")
            cp(db, dt_mismatch_db)
            station_idx, station_dt = h5open(dt_mismatch_db, "r") do f
                read(f["/XcorrP/station_idx"]), read(f["/station/dt"])
            end
            station_dt[station_idx[1]] *= 2
            replace_dataset!(dt_mismatch_db, "/station/dt", station_dt)
            dt_mismatch_run = run_forward(dt_mismatch_db, status0)
            @test !dt_mismatch_run.ok
            @test occursin("XCorr P/S sampling interval mismatch", dt_mismatch_run.stderr)

            missing_gf_db = joinpath(dir, "database_missing_gf.h5")
            cp(db, missing_gf_db)
            h5open(missing_gf_db, "r+") do f
                delete_object(f, "/XcorrP/gf/1/1/1/gf")
            end
            missing_gf_run = run_forward(missing_gf_db, status0)
            @test !missing_gf_run.ok
            @test occursin("DataCache: missing /XcorrP/gf/1/1/1/gf", missing_gf_run.stderr)

            nan_db = joinpath(dir, "database_nan.h5")
            cp(db, nan_db)
            obs_p = h5open(nan_db, "r") do f
                read(f["/XcorrP/obs/1/obs"])
            end
            obs_p[1] = NaN
            replace_dataset!(nan_db, "/XcorrP/obs/1/obs", obs_p)
            nan_run = run_forward(nan_db, status0)
            @test !nan_run.ok
            @test occursin("XCorr observation: non-finite value", nan_run.stderr)

            inf_db = joinpath(dir, "database_inf.h5")
            cp(db, inf_db)
            obs_p = h5open(inf_db, "r") do f
                read(f["/XcorrP/obs/1/obs"])
            end
            obs_p[1] = Inf
            replace_dataset!(inf_db, "/XcorrP/obs/1/obs", obs_p)
            inf_run = run_forward(inf_db, status0)
            @test !inf_run.ok
            @test occursin("XCorr observation: non-finite value", inf_run.stderr)

            window_db = joinpath(dir, "database_window_mismatch.h5")
            cp(db, window_db)
            obs_p, gf_p = h5open(window_db, "r") do f
                read(f["/XcorrP/obs/1/obs"]), read(f["/XcorrP/gf/1/1/1/gf"])
            end
            replace_dataset!(window_db, "/XcorrP/obs/1/obs", obs_p[:, 1:(end - 1)])
            replace_dataset!(window_db, "/XcorrP/gf/1/1/1/gf", gf_p[:, :, 1:(end - 1)])
            window_run = run_forward(window_db, status0)
            @test !window_run.ok
            @test occursin("P/S XCorr window lengths differ", window_run.stderr)

            for path in (status0, short_status, bad_index_status)
                h5open(path, "r") do f
                    @test haskey(f, "/intermediates/XcorrP/cc_max")
                    @test haskey(f, "/intermediates/XcorrS/cc_max")
                    @test !haskey(f, "/intermediates.__tmp__")
                    @test !haskey(f, "/intermediates.__backup__")
                end
            end
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
        cfg = replace(cfg, "maxlag = 3" => "maxlag = 6")
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
            Int32[1, 2, 3],
            Int32(0),
        )
        trials = Search.generate_trials(strat)
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)

        fwd = run_forward(db, status0)
        @test fwd.ok  # must not crash / overrun the clamped allocation

        h5open(db, "r") do f
            global _GFS_CL = Dict(
                (dep, duration) => read(f["/XcorrS/gf/$dep/1/$duration/gf"]) for dep in 1:3 for
                duration in 1:3
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
            global _DURATION_CL = read(f["/trials/duration_idx"])
        end

        n_ent = size(_CC_CL, 2)
        @test n_ent >= 1
        # lag range clamped to ±(501-1)/2 = ±250, not the requested 300
        @test extrema(_LAG_CL) ⊆ (-250:250)
        @test all(-1.0 .<= _CC_CL .<= 1.0)

        max_dcc = 0.0
        for t in 1:N_trials
            dep = Int(_DEP_CL[t])
            gf3 = _GFS_CL[(dep, Int(_DURATION_CL[t]))]
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
