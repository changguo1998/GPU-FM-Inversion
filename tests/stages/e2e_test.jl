# e2e_test.jl — End-to-end pipeline test (input → preprocess → forward → assess → output).
#
# Drives the real `driver.sh` against a freshly generated synthetic event and
# asserts the full single-iteration loop:
#   - driver exits 0
#   - converged after iteration 1 (empty decision file, output produced)
#   - `output.h5` best solution recovers the true source (30, 60, 90) deg
#     (and its |cc_max| complement (30, 60, −90) is equally valid — azimuthal
#     CC has no polarity, so both rake sign nodes share the same misfit)
#   - the XCorrS misfit for the true source is far below random (~0.16 vs ~0.5)
#
# This is the acceptance gate named in AGENTS.md: the XCorr-only pipeline must
# run end-to-end on a synthetic example and reproduce the baseline.
#
# Usage:
#   julia --project=. tests/stages/e2e_test.jl

using Test
using HDF5

include("test_util.jl")

@testset "end-to-end pipeline" begin
    mktempdir() do dir
        make_synthetic(dir; nsta = 6, strike = 30.0, dip = 60.0, rake = 90.0)
        config = write_test_config(dir)

        drv = run_cmd(`bash $(joinpath(PROJECT_ROOT, "driver.sh")) --data-dir $dir`)

        out_path = joinpath(dir, "output.h5")
        @testset "driver run" begin
            @test drv.ok
            @test isfile(out_path)
            @test isfile(joinpath(dir, "status", "status_0.h5"))
            @test isfile(joinpath(dir, ".decision.txt"))
            @test isempty(read(joinpath(dir, ".decision.txt"), String))  # converged
        end

        @testset "convergence" begin
            status0 = joinpath(dir, "status", "status_0.h5")
            h5open(status0, "r") do f
                @test haskey(f, "/intermediates/XcorrS")
                @test haskey(f, "/misfits/XcorrS")
                @test read(f["/trials/N_trials"]) == 151848  # full-space 5° grid
            end
        end

        @testset "solution recovers true source" begin
            h5open(out_path, "r") do f
                s = f["/solution"]
                strike = read(s["strike"])
                dip = read(s["dip"])
                rake = read(s["rake"])
                depth = read(s["depth"])
                misfit = read(s["misfit"])

                # 5° grid: true source (30, 60, 90); rake ±90 complementary
                @test strike == 30.0
                @test dip == 60.0
                @test rake in (90.0, -90.0)
                @test depth == 10.0
                @test misfit < 0.3          # far below random ~0.5
                @test misfit > 0.05

                smry = f["/summary"]
                @test read(smry["total_trials"]) == 151848
                @test read(smry["total_iterations"]) == 1
            end
        end
    end
end
