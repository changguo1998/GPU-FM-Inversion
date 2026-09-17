# e2e_test.jl — End-to-end pipeline test (input → preprocess → forward → assess → output).
#
# Drives the real `driver.sh` on a fresh synthetic event and asserts the single
# iteration loop: driver exits 0, converges after one iteration, and `output.h5`
# recovers the true moment tensor @ 10 km and STF σ=0.2 s. The reported SDR
# may be either nodal-plane representation. Acceptance gate in AGENTS.md /
# doc/roadmap.md: the XCorr P+S pipeline must recover the synthetic source.
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
                @test haskey(f, "/intermediates/XcorrP")
                @test haskey(f, "/misfits/XcorrS")
                @test haskey(f, "/misfits/XcorrP")
                @test read(f["/trials/N_trials"]) == 455544  # 5° grid × 3 durations
            end
        end

        @testset "solution recovers true source" begin
            h5open(out_path, "r") do f
                s = f["/solution"]
                depth = read(s["depth"])
                duration = read(s["duration"])
                duration_idx = read(s["duration_idx"])
                misfit = read(s["misfit"])

                # True source and auxiliary plane represent the same MT.
                mt = read(s["moment_tensor"])
                mt_true = sdr_to_mt(30.0, 60.0, 90.0)
                @test maximum(abs.(mt .- mt_true)) < 1.0e-12
                @test depth == 10.0
                @test duration == 0.2
                @test duration_idx == 2
                @test 0.0 < misfit < 1.0e-3

                smry = f["/summary"]
                @test read(smry["total_trials"]) == 455544
                @test read(smry["total_iterations"]) == 1
            end
        end
    end
end
