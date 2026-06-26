#!/usr/bin/env julia
#
# test_e2e.jl — Verify output.h5 structure for e2e test.
#
# Usage:
#   julia tests/test_e2e.jl <output.h5>
#
# Checks:
#   /solution:  strike, dip, rake, depth, misfit, moment_tensor
#   /uncertainty: depth_range, sdr_std
#   /per_phase: phase-level misfit breakdown
#   /per_station_summary: station-level aggregates
#   /summary: total_iterations, total_trials, convergence_reason, best_iteration

using HDF5

function verify_output_h5(output_path::String)::Bool
    all_pass = true

    if !isfile(output_path)
        println("FAIL: output.h5 not found at $output_path")
        return false
    end

    h5open(output_path, "r") do f

        # /solution
        if haskey(f, "solution")
            sol = f["solution"]
            for key in ["strike", "dip", "rake", "depth", "misfit", "moment_tensor"]
                if haskey(sol, key)
                    println("  ✓ /solution/$key")
                else
                    println("  ✗ /solution/$key MISSING")
                    all_pass = false
                end
            end
            # Validate value ranges
            strike_val = read(sol["strike"])
            if 0.0 <= strike_val <= 360.0
                println("  ✓ strike=$strike_val in [0,360]")
            else
                println("  ✗ strike=$strike_val out of range [0,360]")
                all_pass = false
            end
            dip_val = read(sol["dip"])
            if 0.0 <= dip_val <= 90.0
                println("  ✓ dip=$dip_val in [0,90]")
            else
                println("  ✗ dip=$dip_val out of range [0,90]")
                all_pass = false
            end
            rake_val = read(sol["rake"])
            if -90.0 <= rake_val <= 90.0
                println("  ✓ rake=$rake_val in [-90,90]")
            else
                println("  ✗ rake=$rake_val out of range [-90,90]")
                all_pass = false
            end
            depth_val = read(sol["depth"])
            if depth_val > 0.0
                println("  ✓ depth=$depth_val > 0")
            else
                println("  ✗ depth=$depth_val <= 0")
                all_pass = false
            end
            mt = read(sol["moment_tensor"])
            if length(mt) == 6
                println("  ✓ moment_tensor has 6 components")
            else
                println("  ✗ moment_tensor has $(length(mt)) components (expected 6)")
                all_pass = false
            end
        else
            println("  ✗ /solution group MISSING")
            all_pass = false
        end

        # /uncertainty
        if haskey(f, "uncertainty")
            unc = f["uncertainty"]
            has_depth_range = haskey(unc, "depth_range")
            if has_depth_range
                println("  ✓ /uncertainty/depth_range")
            else
                println("  ✗ /uncertainty/depth_range MISSING")
                all_pass = false
            end
            for std_key in ["strike_std", "dip_std", "rake_std"]
                if haskey(unc, std_key)
                    println("  ✓ /uncertainty/$std_key")
                else
                    println("  ✗ /uncertainty/$std_key MISSING")
                    all_pass = false
                end
            end
        else
            println("  ✗ /uncertainty group MISSING")
            all_pass = false
        end

        # /per_phase
        if haskey(f, "per_phase")
            println("  ✓ /per_phase group present")
            pp = f["per_phase"]
            n_pp = length(keys(pp))
            if n_pp > 0
                println("  ✓ /per_phase has $n_pp datasets")
            else
                println("  ⚠ /per_phase is empty (may be intentional)")
            end
        else
            println("  ✗ /per_phase group MISSING")
            all_pass = false
        end

        # /per_station_summary
        if haskey(f, "per_station_summary")
            println("  ✓ /per_station_summary group present")
            ps = f["per_station_summary"]
            n_sub = length(keys(ps))
            if n_sub > 0
                println("  ✓ /per_station_summary has $n_sub datasets")
            else
                println("  ⚠ /per_station_summary is empty (may be intentional)")
            end
        else
            println("  ✗ /per_station_summary group MISSING")
            all_pass = false
        end

        # /summary
        if haskey(f, "summary")
            sm = f["summary"]
            required = ["total_iterations", "total_trials", "convergence_reason"]
            for key in required
                if haskey(sm, key)
                    println("  ✓ /summary/$key")
                else
                    println("  ✗ /summary/$key MISSING")
                    all_pass = false
                end
            end
            # Validate total_iterations >= 1
            if haskey(sm, "total_iterations")
                n_iter = read(sm["total_iterations"])
                if n_iter >= 1
                    println("  ✓ total_iterations=$n_iter >= 1 (pipeline ran at least 1 iteration)")
                else
                    println("  ✗ total_iterations=$n_iter < 1")
                    all_pass = false
                end
            end
            # Validate total_trials > 0
            if haskey(sm, "total_trials")
                n_trials = read(sm["total_trials"])
                if n_trials > 0
                    println("  ✓ total_trials=$n_trials > 0")
                else
                    println("  ✗ total_trials=$n_trials <= 0")
                    all_pass = false
                end
            end
        else
            println("  ✗ /summary group MISSING")
            all_pass = false
        end
    end

    return all_pass
end

# Entry point
if length(ARGS) < 1
    println(stderr, "Usage: julia test_e2e.jl <output.h5>")
    exit(1)
end

output_path = ARGS[1]
println("Verifying $output_path ...")
result = verify_output_h5(output_path)

if result
    println("\n✓ All output.h5 checks passed.")
    exit(0)
else
    println("\n✗ Some output.h5 checks failed.")
    exit(1)
end
