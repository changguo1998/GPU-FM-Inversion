# output_test.jl — Stage 5 (output) tests.
#
# Chains input → forward → assess on a small trial set, then runs output.jl
# and checks `output.h5` against independent recomputation: `/solution` best
# trial == argmin mean misfit, `/per_phase` cross_correlation == cc_max at
# best, `/uncertainty` fields, `/summary` counts.
#
# Usage:
#   julia --project=. tests/stages/output_test.jl

using Test
using HDF5
using Statistics
using TOML

using IO, Grid

include("test_util.jl")

@testset "output stage" begin
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

        # 108-trial set covering all three STF durations.
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
        trials = Grid.generate_trials(strat)
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)

        @test run_forward(db, status0).ok
        @test run_stage_script(joinpath("scripts", "assess.jl"), [db, status0]).ok

        out = run_stage_script(
            joinpath("scripts", "output.jl"),
            String[];
            env = Dict("DATA_DIR" => dir),
        )
        @testset "run" begin
            @test out.ok
            @test isfile(joinpath(dir, "output.h5"))
        end

        # ── Independent best-trial computation ──
        h5open(db, "r") do f
            global _STRIKE = read(f["/paraspace/strike"])
            global _DIP = read(f["/paraspace/dip"])
            global _RAKE = read(f["/paraspace/rake"])
            global _DEPTH = read(f["/paraspace/depth"])
            global _DURATION = read(f["/paraspace/duration"])
            global _CH_P = String.(read(f["/XcorrP/channel_id"]))
            global _CH_S = String.(read(f["/XcorrS/channel_id"]))
            global _CH = vcat([string(c, ".P") for c in _CH_P], [string(c, ".S") for c in _CH_S])
        end
        h5open(status0, "r") do f
            global _MISFIT_P = read(f["/misfits/XcorrP"])  # [entries × trials]
            global _MISFIT_S = read(f["/misfits/XcorrS"])
            global _TS = read(f["/trials/strike_idx"])
            global _TD = read(f["/trials/dip_idx"])
            global _TR = read(f["/trials/rake_idx"])
            global _TDEP = read(f["/trials/depth_idx"])
            global _TDURATION = read(f["/trials/duration_idx"])
            global _CCMAX_P = read(f["/intermediates/XcorrP/cc_max"])  # [trials × entries]
            global _CCMAX_S = read(f["/intermediates/XcorrS/cc_max"])
            global _CCMAX = hcat(_CCMAX_P, _CCMAX_S)
        end
        mean_p = vec(sum(_MISFIT_P, dims = 1) ./ size(_MISFIT_P, 1))
        mean_s = vec(sum(_MISFIT_S, dims = 1) ./ size(_MISFIT_S, 1))
        mean_m = (mean_p .+ mean_s) ./ 2
        best_idx = argmin(mean_m)

        @testset "output.h5 schema" begin
            h5open(joinpath(dir, "output.h5"), "r") do f
                @testset "/solution" begin
                    s = f["/solution"]
                    @test haskey(s, "strike") && read(s["strike"]) ≈ _STRIKE[_TS[best_idx]]
                    @test read(s["dip"]) ≈ _DIP[_TD[best_idx]]
                    @test read(s["rake"]) ≈ _RAKE[_TR[best_idx]]
                    @test read(s["depth"]) ≈ _DEPTH[_TDEP[best_idx]]
                    @test read(s["duration"]) ≈ _DURATION[_TDURATION[best_idx]]
                    @test read(s["duration_idx"]) == _TDURATION[best_idx]
                    @test read(s["misfit"]) ≈ mean_m[best_idx] atol = 1e-12
                    mt = read(s["moment_tensor"])
                    @test length(mt) == 6
                    @test all(isfinite, mt)
                end

                @testset "/uncertainty" begin
                    u = f["/uncertainty"]
                    @test haskey(u, "strike_std") && haskey(u, "dip_std") && haskey(u, "rake_std")
                    @test haskey(u, "depth_range") && length(read(u["depth_range"])) == 2
                    @test all(v -> isfinite(v), read(u["strike_std"]))
                end

                @testset "/per_phase" begin
                    pp = f["/per_phase"]
                    pids = String.(read(pp["phase_id"]))
                    @test pids == _CH
                    cc = read(pp["cross_correlation"])
                    @test length(cc) == length(pids)
                    # cross_correlation column == cc_max at best trial (per phase)
                    @test cc ≈ _CCMAX[best_idx, :] atol = 1e-12
                    # Both Xcorr modules are represented across the combined P+S phase axis.
                    mpm = read(pp["misfit_per_module"])
                    @test size(mpm, 1) == 2
                    @test size(mpm, 2) == length(pids)
                end

                @testset "/summary" begin
                    s = f["/summary"]
                    @test read(s["total_trials"]) == N_trials
                    @test read(s["total_iterations"]) >= 1
                end
            end
        end

        @testset "result.toml" begin
            txt = joinpath(dir, "result.toml")
            @test isfile(txt)
            parsed = TOML.parsefile(txt)
            @test parsed["format_version"] == 1
            @test parsed["solution"]["duration"] ≈ _DURATION[_TDURATION[best_idx]]
            @test parsed["per_phase"]["phase_id"] == _CH
            @test parsed["per_phase"]["misfit_modules"] == ["XcorrP", "XcorrS"]
            @test length(parsed["per_phase"]["cross_correlation"]) == length(_CH)
            @test parsed["summary"]["total_trials"] == N_trials
        end
    end
end
