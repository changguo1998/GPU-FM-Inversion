# preprocess_test.jl — Stage 2 (preprocess) tests.
#
# Runs `scripts/preprocess.jl` against a prepared `status_0.h5` (strategy
# written via IO.write_strategy — the same file shape input.jl produces),
# then asserts the `/trials` group: index-only Cartesian product of
# strike × dip × rake × depth × freq, all 1-based into /paraspace axes.
#
# Usage:
#   julia --project=. tests/stages/preprocess_test.jl

using Test
using HDF5

using IO, Grid

include("test_util.jl")

@testset "preprocess stage" begin
    mktempdir() do dir
        status_dir = joinpath(dir, "status")
        mkpath(status_dir)
        status0 = joinpath(status_dir, "status_0.h5")

        # Full-space 5° grid (same as input.jl status_0), all depths/freqs.
        g = Grid.default_grid()
        strategy = IO.Strategy(
            g.strike0,
            g.dstrike,
            g.nstrike,
            g.dip0,
            g.ddip,
            g.ndip,
            g.rake0,
            g.drake,
            g.nrake,
            Int32[1, 2, 3],  # /paraspace/depth
            Int32[1],        # /paraspace/frequency
            Int32(0),
        )
        # create file then write strategy into it
        h5open(status0, "w") do f
        end
        IO.write_strategy(status0, strategy)

        r = run_stage_script(
            joinpath("scripts", "preprocess.jl"),
            String[];
            env = Dict("DATA_DIR" => dir),
        )
        @testset "stage run" begin
            @test r.ok
        end

        @testset "/trials" begin
            h5open(status0, "r") do f
                tr = f["/trials"]
                strike_idx = read(tr["strike_idx"])
                dip_idx = read(tr["dip_idx"])
                rake_idx = read(tr["rake_idx"])
                depth_idx = read(tr["depth_idx"])
                freq_idx = read(tr["freq_idx"])
                n_trials = read(tr["N_trials"])
                nd = read(f["/strategy"]["nstrike"])

                expected = Int(72 * 19 * 37 * 3 * 1)
                @test n_trials == expected
                @test length(strike_idx) == expected
                @test length(dip_idx) == expected
                @test length(rake_idx) == expected
                @test length(depth_idx) == expected
                @test length(freq_idx) == expected

                # index ranges: 1-based into /paraspace axes
                @test all(1 .<= strike_idx .<= 72)
                @test all(1 .<= dip_idx .<= 19)
                @test all(1 .<= rake_idx .<= 37)
                @test all(1 .<= depth_idx .<= 3)
                @test all(freq_idx .== 1)

                # uniformity: strike is outermost, each value repeated n/72 times
                @test all(count(==(s), strike_idx) == expected ÷ 72 for s in 1:72)
                @test count(==(1), depth_idx) == expected ÷ 3
                @test all(sort(unique(depth_idx)) == [1, 2, 3])

                # full Cartesian coverage: every combination exactly once
                combos = Set(
                    (strike_idx[i], dip_idx[i], rake_idx[i], depth_idx[i], freq_idx[i]) for
                    i in 1:expected
                )
                @test length(combos) == expected

                @test nd == 72  # sanity: read uses /strategy group intact
            end
        end

        @testset "strategy preserved" begin
            h5open(status0, "r") do f
                st = f["/strategy"]
                @test read(st["nstrike"]) == 72
                @test read(st["depth_indices"]) == [1, 2, 3]
                @test read(st["iteration"]) == 0
            end
        end
    end
end
