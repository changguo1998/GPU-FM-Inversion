#!/usr/bin/env julia
# Test: output stage integration
using Test
using HDF5
using LinearAlgebra
using Statistics

# ── Setup paths ──
include(joinpath(@__DIR__, "..", "src", "solution_comp.jl"))
using .SolutionComp

include(joinpath(@__DIR__, "..", "..", "shared", "HDF5IO.jl", "src", "HDF5IO.jl"))
using .HDF5IO

include(joinpath(@__DIR__, "..", "..", "shared", "MTUtils.jl", "src", "MTUtils.jl"))
using .MTUtils

include(joinpath(@__DIR__, "..", "..", "shared", "AssessUtils.jl", "src", "AssessUtils.jl"))
using .AssessUtils

# ── Test data directory ──
import Base.Filesystem: mktempdir
test_dir = mktempdir(prefix = "output-stage-test-")

@testset "Output stage integration" begin
    println("Test directory: $test_dir")

    # ── 1. Create test database.h5 ──
    @testset "Create test database.h5" begin
        db_path = joinpath(test_dir, "database.h5")

        n_phases = 3
        idx = Index(
            ["NET.ST1.Z.P", "NET.ST1.Z.S", "NET.ST2.Z.P"],
            ["P", "S", "P"],
            Int32[1, 1, 2],
            [10.0, 10.0, 20.0],
            [30.0, 30.0, 60.0],
            fill(Int32(1), n_phases, 3),
        )

        config = Dict(
            "misfit_modules" => ["XCorr", "Polarity", "PSR"],
            "module_weights" => [1.0, 0.5, 0.3],
            "depth_vals" => [5.0, 10.0, 15.0],
            "freq_bands_low" => [0.05],
            "freq_bands_high" => [0.2],
        )

        greens = Dict{String, Dict{Int, Matrix{Float64}}}(
            "NET.ST1.Z.P" => Dict(Int32(1) => randn(100, 6)),
            "NET.ST1.Z.S" => Dict(Int32(1) => randn(100, 6)),
            "NET.ST2.Z.P" => Dict(Int32(1) => randn(100, 6)),
        )

        data = Dict(
            0 => Dict(
                :XCorr => Dict(
                    "NET.ST1.Z.P" => Dict("obs" => randn(100), "gf" => randn(100, 6)),
                    "NET.ST1.Z.S" => Dict("obs" => randn(100), "gf" => randn(100, 6)),
                    "NET.ST2.Z.P" => Dict("obs" => randn(100), "gf" => randn(100, 6)),
                ),
            ),
        )

        write_database(db_path, greens, data, idx, config)
        @test isfile(db_path)
        @test h5exists(db_path, "index/phase_ids")
        @test h5exists(db_path, "config/depth_vals")
        @test h5exists(db_path, "greens/NET.ST1.Z.P/1")
        @test h5exists(db_path, "data/0/XCorr/NET.ST1.Z.P/obs")
        println("  database.h5 created successfully")
    end

    # ── 2. Create test status_2.h5 (converged=1) ──
    status_path = joinpath(test_dir, "status_2.h5")
    run_create_status = @testset "Create test status_2.h5 (converged)" begin
        n_trials = 27

        strategy = Strategy(
            45.0, 10.0, 2,
            30.0, 10.0, 2,
            90.0, 10.0, 2,
            Int32[1], Int32[1],
            Int32[1, 1, 1],
            Int32[1, 1],
            Int32[1, 1],
            [1.0, 0.5, 0.3],
            [45.0, 30.0, 90.0],
            0, 0.12, 2, 1, "user",
            [45.0 30.0 90.0; 46.0 31.0 89.0],
            zeros(Float64, 2, 10),
            [0.3, 0.1, 0.35],
        )

        function make_sdr_grid()
            strikes = Float64[]
            dips = Float64[]
            rakes = Float64[]
            for s in 35.0:10.0:55.0, d in 20.0:10.0:40.0, r in 80.0:10.0:100.0
                push!(strikes, s)
                push!(dips, d)
                push!(rakes, r)
            end
            return strikes, dips, rakes
        end
        strikes, dips, rakes = make_sdr_grid()

        trials = TrialSet(
            strikes, dips, rakes,
            fill(10.0, n_trials),
            fill(Int32(0), n_trials),
            fill(Int32(0), n_trials),
        )

        n_phases = 3
        n_stations = 2
        xcorr = fill(1.0, n_phases, n_trials)
        xcorr[:, 14] .= 0.1
        polarity = fill(1.0, n_stations, n_trials)
        polarity[:, 14] .= 0.0
        psr = fill(0.5, n_stations, n_trials)
        psr[:, 14] .= 0.02

        h5open(status_path, "w") do f
            sgr = HDF5.create_group(f, "strategy")
            write(sgr, "strike0", strategy.strike0)
            write(sgr, "dstrike", strategy.dstrike)
            write(sgr, "nstrike", strategy.nstrike)
            write(sgr, "dip0", strategy.dip0)
            write(sgr, "ddip", strategy.ddip)
            write(sgr, "ndip", strategy.ndip)
            write(sgr, "rake0", strategy.rake0)
            write(sgr, "drake", strategy.drake)
            write(sgr, "nrake", strategy.nrake)
            write(sgr, "depth_indices", strategy.depth_indices)
            write(sgr, "freq_indices", strategy.freq_indices)
            write(sgr, "xcorr_phase_mask", strategy.xcorr_phase_mask)
            write(sgr, "polarity_station_mask", strategy.polarity_station_mask)
            write(sgr, "psr_station_mask", strategy.psr_station_mask)
            write(sgr, "module_weights", strategy.module_weights)
            write(sgr, "best_sdr", strategy.best_sdr)
            write(sgr, "best_depth_index", strategy.best_depth_index)
            write(sgr, "best_misfit", strategy.best_misfit)
            write(sgr, "iteration", strategy.iteration)
            write(sgr, "converged", strategy.converged)
            write(sgr, "convergence_reason", strategy.convergence_reason)
            write(sgr, "freq_accumulated", strategy.freq_accumulated)
            write(sgr, "freq_misfit_curve", strategy.freq_misfit_curve)
            write(sgr, "depth_misfit_accumulated", strategy.depth_misfit_accumulated)

            tgr = HDF5.create_group(f, "trials")
            write(tgr, "strike", trials.strike)
            write(tgr, "dip", trials.dip)
            write(tgr, "rake", trials.rake)
            write(tgr, "depth", trials.depth)
            write(tgr, "depth_idx", trials.depth_idx)
            write(tgr, "freq_idx", trials.freq_idx)

            mgr = HDF5.create_group(f, "misfits")
            write(mgr, "xcorr", xcorr)
            write(mgr, "polarity", polarity)
            write(mgr, "psr", psr)
        end

        @test isfile(status_path)
        @test h5exists(status_path, "strategy/converged")
        @test h5exists(status_path, "trials/strike")
        @test h5exists(status_path, "misfits/xcorr")

        h5open(status_path, "r") do f
            cflag = read(f["strategy/converged"])
            @test cflag == 1
        end
        println("  status_2.h5 created (converged=1)")
    end

    # ── 3. Run output stage ──
    @testset "Run output stage" begin
        db_path = joinpath(test_dir, "database.h5")
        status_dir = test_dir

        cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "src", "output.jl")) $db_path --status-dir $status_dir`
        println("  Running: ", cmd)
        run(cmd)
        println("  output.jl completed")
    end

    # ── 4. Verify output.h5 ──
    @testset "Verify output.h5 contents" begin
        out_path = joinpath(test_dir, "output.h5")
        @test isfile(out_path)

        h5open(out_path, "r") do f
            @test haskey(f, "solution")
            @test haskey(f, "uncertainty")
            @test haskey(f, "per_station")
            @test haskey(f, "summary")

            sol = f["solution"]
            @test haskey(sol, "strike")
            @test haskey(sol, "dip")
            @test haskey(sol, "rake")
            @test haskey(sol, "depth")
            @test haskey(sol, "moment_tensor")
            @test haskey(sol, "misfit")

            strike = read(sol["strike"])
            dip = read(sol["dip"])
            rake = read(sol["rake"])
            mt = read(sol["moment_tensor"])

            @test (35.0 <= strike <= 55.0)
            @test (20.0 <= dip <= 40.0)
            @test (80.0 <= rake <= 100.0)
            @test length(mt) == 6
            @test !all(isnan, mt)
            @test (read(sol["misfit"]) >= 0.0)
            @test (read(sol["misfit"]) < 1.0)

            unc = f["uncertainty"]
            @test haskey(unc, "strike_std")
            @test haskey(unc, "dip_std")
            @test haskey(unc, "rake_std")
            @test haskey(unc, "depth_range")
            dr = read(unc["depth_range"])
            @test length(dr) == 2

            pst = f["per_station"]
            @test haskey(pst, "station_id")
            @test haskey(pst, "phase_type")
            @test haskey(pst, "misfit_per_module")
            @test haskey(pst, "selected")
            @test haskey(pst, "cross_correlation")

            station_ids = read(pst["station_id"])
            @test length(station_ids) >= 1

            sm = f["summary"]
            @test haskey(sm, "total_iterations")
            @test haskey(sm, "total_trials")
            @test haskey(sm, "convergence_reason")
            @test read(sm["total_iterations"]) == 2
            @test read(sm["total_trials"]) == 27
        end

        println("  output.h5 verified: all required groups present")
    end

    # ── 5. Edge case: single trial (clean directory) ──
    @testset "Edge case: single trial" begin
        edge_dir = mktempdir(prefix = "output-edge-")
        status_path = joinpath(edge_dir, "status_0.h5")
        edge_db = joinpath(edge_dir, "database.h5")

        # Create a mini database.h5
        edge_idx = Index(
            ["NET.ST1.Z.P"], ["P"], Int32[1], [10.0], [30.0], fill(Int32(1), 1, 1),
        )
        edge_config = Dict("depth_vals" => [10.0], "module_weights" => [1.0, 1.0, 1.0])
        edge_greens = Dict{String, Dict{Int, Matrix{Float64}}}()
        edge_data = Dict(
            0 => Dict(
                :XCorr => Dict(
                    "NET.ST1.Z.P" => Dict("obs" => zeros(10), "gf" => zeros(10, 6)),
                ),
            ),
        )
        write_database(edge_db, edge_greens, edge_data, edge_idx, edge_config)

        h5open(status_path, "w") do f
            sgr = HDF5.create_group(f, "strategy")
            write(sgr, "strike0", 0.0)
            write(sgr, "dstrike", 0.0)
            write(sgr, "nstrike", Int32(0))
            write(sgr, "dip0", 90.0)
            write(sgr, "ddip", 0.0)
            write(sgr, "ndip", Int32(0))
            write(sgr, "rake0", 0.0)
            write(sgr, "drake", 0.0)
            write(sgr, "nrake", Int32(0))
            write(sgr, "depth_indices", Int32[0])
            write(sgr, "freq_indices", Int32[0])
            write(sgr, "xcorr_phase_mask", Int32[1])
            write(sgr, "polarity_station_mask", Int32[1])
            write(sgr, "psr_station_mask", Int32[1])
            write(sgr, "module_weights", [1.0, 0.5, 0.3])
            write(sgr, "best_sdr", [0.0, 90.0, 0.0])
            write(sgr, "best_depth_index", Int32(0))
            write(sgr, "best_misfit", 0.0)
            write(sgr, "iteration", Int32(0))
            write(sgr, "converged", Int32(1))
            write(sgr, "convergence_reason", "user")
            write(sgr, "freq_accumulated", [0.0 90.0 0.0])
            write(sgr, "freq_misfit_curve", zeros(Float64, 1, 1))
            write(sgr, "depth_misfit_accumulated", [0.0])

            tgr = HDF5.create_group(f, "trials")
            write(tgr, "strike", [0.0])
            write(tgr, "dip", [90.0])
            write(tgr, "rake", [0.0])
            write(tgr, "depth", [10.0])
            write(tgr, "depth_idx", Int32[0])
            write(tgr, "freq_idx", Int32[0])

            mgr = HDF5.create_group(f, "misfits")
            write(mgr, "xcorr", reshape([0.1], 1, 1))
            write(mgr, "polarity", reshape([0.0], 1, 1))
            write(mgr, "psr", reshape([0.0], 1, 1))
        end

        cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "src", "output.jl")) $edge_db --status-dir $edge_dir`
        run(cmd)

        out_path = joinpath(edge_dir, "output.h5")
        @test isfile(out_path)

        h5open(out_path, "r") do f
            sol = f["solution"]
            @test read(sol["strike"]) == 0.0
            @test read(sol["dip"]) == 90.0
            @test read(sol["rake"]) == 0.0
            mt = read(sol["moment_tensor"])
            @test length(mt) == 6
            # Pure double-couple: [0,0,0,1,0,0]
            @test isapprox(mt[1], 0.0, atol=1e-12)
            @test isapprox(mt[2], 0.0, atol=1e-12)
            @test isapprox(mt[3], 0.0, atol=1e-12)
            @test isapprox(mt[4], 1.0, atol=1e-12)
            @test isapprox(mt[5], 0.0, atol=1e-12)
            @test isapprox(mt[6], 0.0, atol=1e-12)
        end

        rm(edge_dir; recursive=true, force=true)
        println("  Single trial edge case: PASS")
    end
end

println("\nAll output stage integration tests passed!")

# Cleanup
rm(test_dir; recursive=true, force=true)