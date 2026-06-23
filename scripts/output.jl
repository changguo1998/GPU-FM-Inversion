#!/usr/bin/env julia
# output.jl — Final Solution Compilation Stage

using Printf
using Statistics: std

# ── Load shared modules ──
SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "..", "shared", "io", "src", "IO.jl"))
using .IO

include(joinpath(SCRIPT_DIR, "..", "shared", "mt", "src", "MT.jl"))
using .MT

include(joinpath(SCRIPT_DIR, "..", "shared", "aggregate", "src", "Aggregate.jl"))
using .Aggregate

# ═══════════════════════════════════════════════════════════════════════════════
# Solution compilation functions (inlined from solution_comp.jl)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_depth_range(
        depth_vals::AbstractVector{Float64},
        depth_misfit_vec::AbstractVector{Float64},
        tolerance::Float64=0.05,
    ) -> Vector{Float64}

Compute depth range by applying tolerance to per-depth misfit accumulated by assess.jl.

# Returns
- `[min_depth, max_depth]` — depths within `tolerance` fraction of the best misfit.
"""
function compute_depth_range(
    depth_vals::AbstractVector{Float64},
    depth_misfit_vec::AbstractVector{Float64};
    tolerance::Float64 = 0.05,
)
    if length(depth_misfit_vec) == 0
        return [NaN, NaN]
    end

    valid = filter(!isnan, depth_misfit_vec)
    if isempty(valid)
        return [NaN, NaN]
    end
    best_misfit = minimum(valid)
    threshold = best_misfit * (1.0 + tolerance)

    valid_depths = Float64[]
    for i in eachindex(depth_misfit_vec)
        if !isnan(depth_misfit_vec[i]) && depth_misfit_vec[i] <= threshold
            push!(valid_depths, depth_vals[i])
        end
    end

    if isempty(valid_depths)
        return [NaN, NaN]
    end

    return [minimum(valid_depths), maximum(valid_depths)]
end

"""
    compute_sdr_std(freq_accumulated::AbstractMatrix{Float64}) -> (Float64, Float64, Float64)

Compute standard deviation of strike, dip, rake across frequency bands.

`freq_accumulated` has shape `[N_frequencies, 3]` where each row is `[strike, dip, rake]`.

# Returns
- `(strike_std, dip_std, rake_std)`
"""
function compute_sdr_std(freq_accumulated::AbstractMatrix{Float64})
    if size(freq_accumulated, 1) <= 1
        return (NaN, NaN, NaN)
    end
    s_std = std(freq_accumulated[:, 1])
    d_std = std(freq_accumulated[:, 2])
    r_std = std(freq_accumulated[:, 3])
    return (s_std, d_std, r_std)
end

"""
    compile_solution(
        strategy, trials, misfits, index, config;
        synthesize_waveforms::Bool=false,
    ) -> NamedTuple

Compile final focal mechanism solution from converged pipeline results.
"""
function compile_solution(
    strategy,
    trials,
    misfits,
    index,
    config;
    synthesize_waveforms::Bool = false,
)
    n_trials = length(trials.strike)
    n_phases = length(index.phase_ids)
    n_stations = length(unique(index.station_idx))

    # ── 1. Re-aggregate misfits to find best trial ──
    module_weights = strategy.module_weights
    if length(module_weights) < 3
        module_weights = [module_weights; zeros(Float64, 3 - length(module_weights))]
    end

    xcorr_data = get(misfits, :xcorr, zeros(Float64, n_phases, n_trials))
    polarity_data = get(misfits, :polarity, zeros(Float64, n_stations, n_trials))
    psr_data = get(misfits, :psr, zeros(Float64, n_stations, n_trials))

    xcorr_mask = strategy.xcorr_phase_mask
    if length(xcorr_mask) < n_phases
        xcorr_mask = ones(Int32, n_phases)
    end
    pol_mask = strategy.polarity_channel_mask
    if length(pol_mask) < n_stations
        pol_mask = ones(Int32, n_stations)
    end
    psr_mask = strategy.psr_channel_mask
    if length(psr_mask) < n_stations
        psr_mask = ones(Int32, n_stations)
    end

    # Convert Int32 masks to Bool
    xcorr_bool = Vector{Bool}(xcorr_mask .== Int32(1))
    pol_bool = Vector{Bool}(pol_mask .== Int32(1))
    psr_bool = Vector{Bool}(psr_mask .== Int32(1))

    total, best_idx, per_module_scores = Aggregate.aggregate_misfits(
        xcorr_data, polarity_data, psr_data,
        xcorr_bool, pol_bool, psr_bool,
        module_weights,
    )

    best_strike = trials.strike[best_idx]
    best_dip = trials.dip[best_idx]
    best_rake = trials.rake[best_idx]
    best_depth = trials.depth[best_idx]
    best_misfit = total[best_idx]

    # ── 2. SDR → MT conversion ──
    mt = MT.sdr_to_mt(best_strike, best_dip, best_rake)

    solution = Dict{String, Any}(
        "strike" => best_strike,
        "dip" => best_dip,
        "rake" => best_rake,
        "depth" => best_depth,
        "moment_tensor" => mt,
        "misfit" => best_misfit,
    )

    # ── 3. Frequency uncertainty ──
    freq_accumulated = getfield(strategy, :freq_accumulated)
    strike_std_val, dip_std_val, rake_std_val = compute_sdr_std(freq_accumulated)
    freq_misfit_curve = getfield(strategy, :freq_misfit_curve)

    # ── 4. Depth range ──
    depth_vals_from_config = get(config, "depth_vals", Float64[])
    if isempty(depth_vals_from_config) && haskey(config, "depth")
        dcfg = config["depth"]
        if dcfg isa Vector
            depth_vals_from_config = dcfg
        elseif dcfg isa Dict && haskey(dcfg, "vals")
            depth_vals_from_config = dcfg["vals"]
        end
    end
    depth_misfit_vec = getfield(strategy, :depth_misfit_accumulated)
    depth_range = compute_depth_range(depth_vals_from_config, depth_misfit_vec)

    uncertainty = Dict{String, Any}(
        "strike_std" => strike_std_val,
        "dip_std" => dip_std_val,
        "rake_std" => rake_std_val,
        "depth_range" => depth_range,
        "freq_test_misfit_curve" => freq_misfit_curve,
    )

    # ── 5. Per-phase breakdown ──
    phase_ids = index.phase_ids
    phase_types = index.phase_type
    station_indices = index.station_idx
    n_stations = length(unique(station_indices))

    # Build station and channel identifiers from phase_ids
    station_ids = [split(pid, ".")[1] * "." * split(pid, ".")[2] for pid in phase_ids]
    channel_ids = [join(split(pid, ".")[1:3], ".") for pid in phase_ids]

    # Extract per-phase misfits at best trial
    misfit_per_module = zeros(Float64, 3, n_phases)
    xc = get(misfits, :xcorr, zeros(Float64, n_phases, n_trials))
    pol = get(misfits, :polarity, zeros(Float64, n_stations, n_trials))
    psr = get(misfits, :psr, zeros(Float64, n_stations, n_trials))

    for ph in 1:n_phases
        misfit_per_module[1, ph] = xc[ph, best_idx]
    end
    for ph in 1:n_phases
        si = station_indices[ph]
        if 1 <= si <= size(pol, 1)
            misfit_per_module[2, ph] = pol[si, best_idx]
        end
        if 1 <= si <= size(psr, 1)
            misfit_per_module[3, ph] = psr[si, best_idx]
        end
    end

    selected = xcorr_mask
    if length(selected) < n_phases
        selected = ones(Int32, n_phases)
    end

    cross_correlation = [1.0 - misfit_per_module[1, ph] for ph in 1:n_phases]

    per_phase = Dict{String, Any}(
        "phase_id" => phase_ids,
        "channel_id" => channel_ids,
        "station_id" => station_ids,
        "phase_type" => phase_types,
        "misfit_per_module" => misfit_per_module,
        "selected" => selected,
        "cross_correlation" => cross_correlation,
    )

    # ── 6. Per-station summary ───────────────────────────────────────────────────
    unique_stations = unique(station_ids)
    n_unique = length(unique_stations)

    sta_summary_station_ids = String[]
    sta_summary_n_channels = Int32[]
    sta_summary_n_phases = Int32[]
    sta_summary_mean_cc = Float64[]
    sta_summary_pol_match = Int32[]
    sta_summary_misfit_total = Float64[]

    for sta in unique_stations
        idx_in_sta = findall(s -> s == sta, station_ids)
        n_ph_in_sta = length(idx_in_sta)

        channels_in_sta = unique([channel_ids[i] for i in idx_in_sta])
        n_ch = length(channels_in_sta)

        cc_vals = [cross_correlation[i] for i in idx_in_sta]
        mean_cc = sum(cc_vals) / length(cc_vals)

        pol_match = 0
        for i in idx_in_sta
            si = station_indices[i]
            if 1 <= si <= size(pol, 1) && pol[si, best_idx] == 0.0
                pol_match += 1
            end
        end

        total_misfit = 0.0
        for i in idx_in_sta
            for m in 1:3
                v = misfit_per_module[m, i]
                if !isnan(v)
                    total_misfit += v
                end
            end
        end

        push!(sta_summary_station_ids, sta)
        push!(sta_summary_n_channels, Int32(n_ch))
        push!(sta_summary_n_phases, Int32(n_ph_in_sta))
        push!(sta_summary_mean_cc, mean_cc)
        push!(sta_summary_pol_match, Int32(pol_match))
        push!(sta_summary_misfit_total, total_misfit)
    end

    per_station_summary = Dict{String, Any}(
        "station_id" => sta_summary_station_ids,
        "n_channels" => sta_summary_n_channels,
        "n_phases" => sta_summary_n_phases,
        "mean_cross_correlation" => sta_summary_mean_cc,
        "polarity_match" => sta_summary_pol_match,
        "misfit_total" => sta_summary_misfit_total,
    )

    # ── 7. Summary ──
    convergence_reason = getfield(strategy, :convergence_reason)
    total_iterations = getfield(strategy, :iteration)

    summary = Dict{String, Any}(
        "total_iterations" => total_iterations,
        "total_trials" => n_trials,
        "convergence_reason" => convergence_reason,
    )

    return (solution = solution, uncertainty = uncertainty,
            per_phase = per_phase, per_station_summary = per_station_summary, summary = summary)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CLI parsing
# ═══════════════════════════════════════════════════════════════════════════════

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
        println(stderr, "Usage: julia scripts/output.jl <database.h5> [--status-dir <dir>] [--waveforms]")
        exit(1)
    end

    if !isfile(db_path)
        println(stderr, "ERROR: database file not found: $db_path")
        exit(1)
    end

    return (db_path, status_dir, synthesize_waveforms)
end

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

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

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
    strategy = IO.read_strategy(latest_status)
    trials = IO.read_trials(latest_status)
    misfits = IO.read_misfits(latest_status)

    println("[output] Reading index and config from $db_path")
    index = IO.read_index(db_path)
    config = IO.read_config(db_path)

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
            if IO.h5exists(db_path, greens_path)
                gf = IO.read_greens(db_path, ph_id, best_depth_idx)
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
            solgr = HDF5.create_group(f, "solution")
            for (k, v) in solution_wf
                HDF5.write(solgr, k isa Symbol ? string(k) : k, v)
            end
            ungr = HDF5.create_group(f, "uncertainty")
            for (k, v) in uncertainty_wf
                HDF5.write(ungr, k isa Symbol ? string(k) : k, v)
            end
            pphgr = HDF5.create_group(f, "per_phase")
            for (k, v) in per_phase_wf
                HDF5.write(pphgr, k isa Symbol ? string(k) : k, v)
            end
            pstgr = HDF5.create_group(f, "per_station_summary")
            for (k, v) in per_station_summary_wf
                HDF5.write(pstgr, k isa Symbol ? string(k) : k, v)
            end
            smgr = HDF5.create_group(f, "summary")
            for (k, v) in summary_wf
                HDF5.write(smgr, k isa Symbol ? string(k) : k, v)
            end
            wfgr = HDF5.create_group(f, "waveforms")
            for (ph_id, wf) in waveforms
                HDF5.write(wfgr, ph_id, wf)
            end
        end
    else
        IO.write_output(out_path, result.solution, result.uncertainty,
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

if !isinteractive()
    exit(main(ARGS))
end