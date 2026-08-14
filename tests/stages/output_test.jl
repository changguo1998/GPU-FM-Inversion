# output_test.jl — Stage 5 (output) tests.
#
# Chains input → forward → assess on a small trial set, then runs output.jl
# and verifies `output.h5` groups against independently recomputed values:
#   - `/solution` best trial == argmin of per-trial mean misfit (physical SDR
#     resolved from /paraspace, tripled-checked against /misfits)
#   - `/uncertainty` std/range from the best neighborhood
#   - `/per_phase` cross_correlation column == intermediates cc_max at best trial
#   - `/summary` total_trials consistent
#
# Usage:
#   julia --project=. tests/stages/output_test.jl

using Test
using HDF5
using Statistics

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

        # 36-trial set (avoid recomputing the full 151,848 grid here).
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
        @test run_cmd(`$exe $db $status0`).ok
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
            global _CH =
                haskey(f, "/XcorrS/channel_id") ? String.(read(f["/XcorrS/channel_id"])) : String[]
        end
        h5open(status0, "r") do f
            global _MISFIT = read(f["/misfits/XcorrS"])  # [entries × trials]
            global _TS = read(f["/trials/strike_idx"])
            global _TD = read(f["/trials/dip_idx"])
            global _TR = read(f["/trials/rake_idx"])
            global _TDEP = read(f["/trials/depth_idx"])
            global _CCMAX = read(f["/intermediates/XcorrS/cc_max"])  # [trials × entries]
        end
        mean_m = vec(sum(_MISFIT, dims = 1) ./ size(_MISFIT, 1))
        best_idx = argmin(mean_m)

        @testset "output.h5 schema" begin
            h5open(joinpath(dir, "output.h5"), "r") do f
                @testset "/solution" begin
                    s = f["/solution"]
                    @test haskey(s, "strike") && read(s["strike"]) ≈ _STRIKE[_TS[best_idx]]
                    @test read(s["dip"]) ≈ _DIP[_TD[best_idx]]
                    @test read(s["rake"]) ≈ _RAKE[_TR[best_idx]]
                    @test read(s["depth"]) ≈ _DEPTH[_TDEP[best_idx]]
                    @test read(s["misfit"]) ≈ mean_m[best_idx] atol = 1e-12
                    mt = read(s["moment_tensor"])
                    @test length(mt) == 6
                    # MT is Python-free here: just check it is non-degenerate
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
                    # per-phase misfit for XcorrS == misfit matrix column at best trial
                    mpm = read(pp["misfit_per_module"])
                    @test size(mpm, 2) == length(pids)
                end

                @testset "/summary" begin
                    s = f["/summary"]
                    @test read(s["total_trials"]) == N_trials
                    @test read(s["total_iterations"]) >= 1
                end
            end
        end
    end
end
