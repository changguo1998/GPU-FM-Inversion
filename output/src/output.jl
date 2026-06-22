#!/usr/bin/env julia
# output.jl — Final Solution Compilation Stage

using Printf

# ── Resolve project-relative paths ──
script_dir = @__DIR__
shared_dir = joinpath(script_dir, "..", "..", "shared")

# Load shared modules
include(joinpath(shared_dir, "HDF5IO.jl", "src", "HDF5IO.jl"))
using .HDF5IO

include(joinpath(shared_dir, "MTUtils.jl", "src", "MTUtils.jl"))
using .MTUtils

include(joinpath(shared_dir, "AssessUtils.jl", "src", "AssessUtils.jl"))
using .AssessUtils

# Load solution compilation
include(joinpath(@__DIR__, "solution_comp.jl"))
using .SolutionComp

# ── CLI parsing ──
function parse_args(args::Vector{String})
    db_path = ""
    status_dir = "."
    synthesize_waveforms = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--status-dir"
            i += 1
            status_dir = args[i]
        elseif arg == "--waveforms"
            synthesize_waveforms = true
        elseif !startswith(arg, "--")
            db_path = arg
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if isempty(db_path)
        println(stderr, "Usage: julia --project=output output/src/output.jl <database.h5> [--status-dir <dir>] [--waveforms]")
        exit(1)
    end

    if !isfile(db_path)
        println(stderr, "ERROR: database file not found: $db_path")
        exit(1)
    end

    return (db_path, status_dir, synthesize_waveforms)
end

# ── Status file discovery ──
function find_latest_status(status_dir::String)
    pattern = r"^status_(\d+)\.h5$"
    max_n = -1
    latest = ""

    dir_entries = readdir(status_dir; join=true)
    for entry in dir_entries
        fname = basename(entry)
        m = match(pattern, fname)
        if m !== nothing
            n = parse(Int, m.captures[1])
            if n > max_n
                max_n = n
                latest = entry
            end
        end
    end

    if max_n == -1
        println(stderr, "ERROR: no status files found in $status_dir")
        exit(1)
    end

    return (latest, max_n)
end

# ── Main ──
function main(args::Vector{String})
    db_path, status_dir, synthesize_waveforms = parse_args(args)
    println("[output] Starting solution compilation")
    println("[output] Database: $(abspath(db_path))")
    println("[output] Status dir: $(abspath(status_dir))")

    # 1. Discover status files
    latest_status, max_n = find_latest_status(status_dir)
    println("[output] Latest status: $latest_status (iteration $max_n)")

    # 2. Read inputs
    println("[output] Reading final strategy, trials, misfits from $latest_status")
    strategy = read_strategy(latest_status)
    trials = read_trials(latest_status)
    misfits = read_misfits(latest_status)

    println("[output] Reading index and config from $db_path")
    index = read_index(db_path)
    config = read_config(db_path)

    println("[output] strategy.converged = $(strategy.converged)")
    if strategy.converged == 0
        @warn "Final status file has converged=0 — proceeding anyway"
    end

    # 3. Compile solution
    println("[output] Compiling final solution...")
    result = compile_solution(
        strategy, trials, misfits, index, config;
        synthesize_waveforms = false,
    )

    # 4. Waveform synthesis (optional)
    waveforms = nothing
    if synthesize_waveforms
        println("[output] Synthesizing waveforms (GF × MT)...")
        best_mt = result.solution["moment_tensor"]
        best_depth_idx = strategy.best_depth_index
        waveforms = Dict{String, Vector{Float64}}()

        for ph_id in index.phase_ids
            greens_path = "greens/$ph_id/$best_depth_idx"
            if h5exists(db_path, greens_path)
                gf = read_greens(db_path, ph_id, best_depth_idx)
                synthetic = gf * best_mt
                waveforms[ph_id] = synthetic
            else
                @warn "No GF found for phase $ph_id at depth $best_depth_idx"
            end
        end
    end

    # 5. Write output.h5
    out_path = joinpath(status_dir, "output.h5")
    println("[output] Writing $out_path ...")

    if waveforms !== nothing
        solution_wf = copy(result.solution)
        uncertainty_wf = copy(result.uncertainty)
        per_phase_wf = copy(result.per_phase)
        per_station_summary_wf = copy(result.per_station_summary)
        summary_wf = copy(result.summary)

        h5open(out_path, "cw") do f
            # /solution
            solgr = HDF5.create_group(f, "solution")
            for (k, v) in solution_wf
                HDF5.write(solgr, k isa Symbol ? string(k) : k, v)
            end

            # /uncertainty
            ungr = HDF5.create_group(f, "uncertainty")
            for (k, v) in uncertainty_wf
                HDF5.write(ungr, k isa Symbol ? string(k) : k, v)
            end

            # /per_phase
            pphgr = HDF5.create_group(f, "per_phase")
            for (k, v) in per_phase_wf
                HDF5.write(pphgr, k isa Symbol ? string(k) : k, v)
            end

            # /per_station_summary
            pstgr = HDF5.create_group(f, "per_station_summary")
            for (k, v) in per_station_summary_wf
                HDF5.write(pstgr, k isa Symbol ? string(k) : k, v)
            end

            # /summary
            smgr = HDF5.create_group(f, "summary")
            for (k, v) in summary_wf
                HDF5.write(smgr, k isa Symbol ? string(k) : k, v)
            end

            # /waveforms
            wfgr = HDF5.create_group(f, "waveforms")
            for (ph_id, wf) in waveforms
                HDF5.write(wfgr, ph_id, wf)
            end
        end
    else
        write_output(out_path, result.solution, result.uncertainty,
                     result.per_phase, result.per_station_summary, result.summary)
    end

    # 6. Summary
    println("\n" * "="^60)
    println(" OUTPUT COMPLETE")
    println("="^60)
    println("  Strike    : $(round(result.solution["strike"]; digits=2))°")
    println("  Dip       : $(round(result.solution["dip"]; digits=2))°")
    println("  Rake      : $(round(result.solution["rake"]; digits=2))°")
    println("  Depth     : $(round(result.solution["depth"]; digits=2)) km")
    println("  Misfit    : $(round(result.solution["misfit"]; digits=6))")
    println("  Iterations: $(result.summary["total_iterations"])")
    println("  Trials    : $(result.summary["total_trials"])")
    println("  Reason    : $(result.summary["convergence_reason"])")
    println("  Output    : $(abspath(out_path))")
    println("="^60)

    return 0
end

# ── Entry point ──
if !isinteractive()
    exit(main(ARGS))
end