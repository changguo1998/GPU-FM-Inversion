# assess_test.jl — Stage 4 (assess) tests.
#
# Chains input → forward on a small hand-built trial set, then runs assess.jl
# and verifies:
#   - `/misfits/XcorrS` == 1 − cc_max from the intermediates (extract correct)
#   - misfit matrix shape [N_entries × N_trials]
#   - convergence decision file written (empty = converged) when DATA_DIR set
#   - assess does not touch `/strategy` or `/trials`
#
# Usage:
#   julia --project=. tests/stages/assess_test.jl

using Test
using HDF5

using IO, Grid

include("test_util.jl")

@testset "assess stage" begin
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

        # Small trial set (36 trials) so the whole chain stays fast.
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
        @test fwd.ok

        # ── Assess with DATA_DIR set (exercises the decision-file path) ──
        a = run_stage_script(
            joinpath("scripts", "assess.jl"),
            [db, status0];
            env = Dict("DATA_DIR" => dir),
        )
        @testset "run" begin
            @test a.ok
            @test isfile(joinpath(dir, ".decision.txt"))
            @test isempty(read(joinpath(dir, ".decision.txt"), String))  # converged
        end

        @testset "misfits written" begin
            h5open(status0, "r") do f
                @test haskey(f, "/misfits")
                @test haskey(f["/misfits"], "XcorrS")
                m = read(f["/misfits/XcorrS"])
                n_ent = size(m, 1)  # entries; channel_id may be absent when 0 entries
                @test size(m) == (n_ent, N_trials)
                @test all(0.0 .<= m .<= 1.0)

                # extract: misfit = 1 − cc_max
                cc = read(f["/intermediates/XcorrS/cc_max"])  # (N_trials, N_entries)
                expect = 1.0 .- permutedims(cc)
                @test m ≈ expect atol = 1e-12
            end
            h5open(db, "r") do f
                @test string.(read(f["/config/misfit_modules"])) == ["XcorrS"]
            end
        end

        @testset "strategy & trials untouched" begin
            h5open(status0, "r") do f
                @test read(f["/strategy/iteration"]) == 0
                @test read(f["/strategy/nstrike"]) == 72  # input's full grid, untouched by assess
                @test length(read(f["/trials/strike_idx"])) == N_trials
            end
        end
    end
end
