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
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

# Parse positional db path and flags
status_dir, synthesize_waveforms, db_path = let sd = ".", sw = false, dp = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--status-dir"
            i += 1; sd = ARGS[i]
        elseif a == "--waveforms"
            sw = true
        elseif !startswith(a, "--")
            dp = a
        else
            println(stderr, "Unknown argument: $a")
            exit(1)
        end
        i += 1
    end
    (sd, sw, dp)
end

if isempty(db_path)
    println(stderr, "Usage: julia scripts/output.jl <database.h5> [--status-dir <dir>] [--waveforms]")
    exit(1)
end
if !isfile(db_path)
    println(stderr, "ERROR: database file not found: $db_path")
    exit(1)
end

println("[output] Starting solution compilation")
println("[output] Database: $(abspath(db_path))")
println("[output] Status dir: $(abspath(status_dir))")

# ═══════════════════════════════════════════════════════════════════════════════
# Find latest status file
# ═══════════════════════════════════════════════════════════════════════════════

latest_status, max_n = IO.find_latest_status(status_dir)
println("[output] Latest status: $latest_status (iteration $max_n)")

# ═══════════════════════════════════════════════════════════════════════════════
# Read inputs
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# Compile solution
# ═══════════════════════════════════════════════════════════════════════════════

n_trials = length(trials.strike)
n_phases = length(index.phase_ids)

# 1. Re-aggregate misfits
module_weights = strategy.module_weights
if length(module_weights) < 3
    module_weights = [module_weights; zeros(Float64, 3 - length(module_weights))]
end

xc = get(misfits, :xcorr, zeros(Float64, n_phases, n_trials))
pol = get(misfits, :polarity, zeros(Float64, length(unique(index.station_idx)), n_trials))
psr = get(misfits, :psr, zeros(Float64, length(unique(index.station_idx)), n_trials))

xmask = strategy.xcorr_phase_mask
polmask = strategy.polarity_channel_mask
psrmask = strategy.psr_channel_mask
n_stations = size(pol, 1)

if length(xmask) < n_phases; xmask = ones(Int32, n_phases); end
if length(polmask) < n_stations; polmask = ones(Int32, n_stations); end
if length(psrmask) < n_stations; psrmask = ones(Int32, n_stations); end

xcorr_bool = Vector{Bool}(xmask .== Int32(1))
pol_bool = Vector{Bool}(polmask .== Int32(1))
psr_bool = Vector{Bool}(psrmask .== Int32(1))

total, best_idx, per_module = Aggregate.aggregate_misfits(
    xc, pol, psr, xcorr_bool, pol_bool, psr_bool, module_weights)

best_strike = trials.strike[best_idx]
best_dip = trials.dip[best_idx]
best_rake = trials.rake[best_idx]
best_depth = trials.depth[best_idx]
best_misfit = total[best_idx]

# 2. SDR → MT
mt = MT.sdr_to_mt(best_strike, best_dip, best_rake)

solution = Dict{String, Any}(
    "strike" => best_strike, "dip" => best_dip, "rake" => best_rake,
    "depth" => best_depth, "moment_tensor" => mt, "misfit" => best_misfit,
)

# 3. Frequency uncertainty
freq_accumulated = getfield(strategy, :freq_accumulated)
strike_std_val, dip_std_val, rake_std_val = Aggregate.compute_sdr_std(freq_accumulated)
freq_misfit_curve = getfield(strategy, :freq_misfit_curve)

# 4. Depth range
depth_vals_from_config = get(config, "depth_vals", Float64[])
if isempty(depth_vals_from_config) && haskey(config, "depth")
    dcfg = config["depth"]
    depth_vals_from_config = if dcfg isa Vector; dcfg
    elseif dcfg isa Dict && haskey(dcfg, "vals"); dcfg["vals"]
    else depth_vals_from_config end
end
depth_misfit_vec = getfield(strategy, :depth_misfit_accumulated)
depth_range = Aggregate.compute_depth_range(depth_vals_from_config, depth_misfit_vec)

uncertainty = Dict{String, Any}(
    "strike_std" => strike_std_val, "dip_std" => dip_std_val,
    "rake_std" => rake_std_val, "depth_range" => depth_range,
    "freq_test_misfit_curve" => freq_misfit_curve,
)

# 5. Per-phase breakdown
phase_ids = index.phase_ids
phase_types = index.phase_type
station_indices = index.station_idx

station_ids = [join(split(pid, ".")[1:2], ".") for pid in phase_ids]
channel_ids = [join(split(pid, ".")[1:3], ".") for pid in phase_ids]

misfit_per_module = zeros(Float64, 3, n_phases)
for ph in 1:n_phases
    misfit_per_module[1, ph] = xc[ph, best_idx]
end
for ph in 1:n_phases
    si = station_indices[ph]
    if 1 <= si <= size(pol, 1); misfit_per_module[2, ph] = pol[si, best_idx]; end
    if 1 <= si <= size(psr, 1); misfit_per_module[3, ph] = psr[si, best_idx]; end
end

selected = length(xmask) < n_phases ? ones(Int32, n_phases) : xmask
cross_correlation = [1.0 - misfit_per_module[1, ph] for ph in 1:n_phases]

per_phase = Dict{String, Any}(
    "phase_id" => phase_ids, "channel_id" => channel_ids,
    "station_id" => station_ids, "phase_type" => phase_types,
    "misfit_per_module" => misfit_per_module, "selected" => selected,
    "cross_correlation" => cross_correlation,
)

# 6. Per-station summary
unique_stations = unique(station_ids)
sta_ids = String[]
sta_n_ch = Int32[]
sta_n_ph = Int32[]
sta_cc = Float64[]
sta_pol = Int32[]
sta_mis = Float64[]

for sta in unique_stations
    idx_in_sta = findall(s -> s == sta, station_ids)
    n_ph_sta = length(idx_in_sta)
    n_ch = length(unique([channel_ids[i] for i in idx_in_sta]))
    cc_vals = [cross_correlation[i] for i in idx_in_sta]
    mean_cc = sum(cc_vals) / length(cc_vals)
    pol_match = 0
    for i in idx_in_sta
        si = station_indices[i]
        if 1 <= si <= size(pol, 1) && pol[si, best_idx] == 0.0; pol_match += 1; end
    end
    total_misfit = 0.0
    for i in idx_in_sta
        for m in 1:3
            v = misfit_per_module[m, i]
            !isnan(v) && (total_misfit += v)
        end
    end
    push!(sta_ids, sta)
    push!(sta_n_ch, Int32(n_ch))
    push!(sta_n_ph, Int32(n_ph_sta))
    push!(sta_cc, mean_cc)
    push!(sta_pol, Int32(pol_match))
    push!(sta_mis, total_misfit)
end

per_station_summary = Dict{String, Any}(
    "station_id" => sta_ids, "n_channels" => sta_n_ch,
    "n_phases" => sta_n_ph, "mean_cross_correlation" => sta_cc,
    "polarity_match" => sta_pol, "misfit_total" => sta_mis,
)

# 7. Summary
summary = Dict{String, Any}(
    "total_iterations" => getfield(strategy, :iteration),
    "total_trials" => n_trials,
    "convergence_reason" => getfield(strategy, :convergence_reason),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Waveform synthesis (optional)
# ═══════════════════════════════════════════════════════════════════════════════

waveforms = nothing
if synthesize_waveforms
    println("[output] Synthesizing waveforms (GF × MT)...")
    best_mt = solution["moment_tensor"]
    best_depth_idx = strategy.best_depth_index
    waveforms = Dict{String, Vector{Float64}}()

    for ph_id in phase_ids
        gp = "greens/$ph_id/$best_depth_idx"
        if IO.h5exists(db_path, gp)
            gf = IO.read_greens(db_path, ph_id, best_depth_idx)
            waveforms[ph_id] = gf * best_mt
        else
            @warn "No GF found for phase $ph_id at depth $best_depth_idx"
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Write output.h5
# ═══════════════════════════════════════════════════════════════════════════════

out_path = joinpath(status_dir, "output.h5")
println("[output] Writing $out_path ...")

if waveforms !== nothing
    h5open(out_path, "cw") do f
        for (grp_name, grp_dict) in [
            ("solution", solution),
            ("uncertainty", uncertainty),
            ("per_phase", per_phase),
            ("per_station_summary", per_station_summary),
            ("summary", summary),
        ]
            gr = HDF5.create_group(f, grp_name)
            for (k, v) in grp_dict
                HDF5.write(gr, k isa Symbol ? string(k) : k, v)
            end
        end
        wfgr = HDF5.create_group(f, "waveforms")
        for (ph_id, wf) in waveforms
            HDF5.write(wfgr, ph_id, wf)
        end
    end
else
    IO.write_output(out_path, solution, uncertainty, per_phase, per_station_summary, summary)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^60)
println(" OUTPUT COMPLETE")
println("="^60)
@printf("  Strike    : %.2f°\n", solution["strike"])
@printf("  Dip       : %.2f°\n", solution["dip"])
@printf("  Rake      : %.2f°\n", solution["rake"])
@printf("  Depth     : %.2f km\n", solution["depth"])
@printf("  Misfit    : %.6f\n", solution["misfit"])
println("  Iterations: $(summary["total_iterations"])")
println("  Trials    : $(summary["total_trials"])")
println("  Reason    : $(summary["convergence_reason"])")
println("  Output    : $(abspath(out_path))")
println("="^60)