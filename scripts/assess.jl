#!/usr/bin/env julia

using HDF5

# ── Load shared modules ────────────────────────────────────────────────────────
SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "..", "shared", "io", "src", "IO.jl"))
using .IO
include(joinpath(SCRIPT_DIR, "..", "shared", "aggregate", "src", "Aggregate.jl"))
using .Aggregate: aggregate_misfits
include(joinpath(SCRIPT_DIR, "..", "shared", "grid", "src", "Grid.jl"))
using .Grid: refine_strategy, prompt_operator, TrialResult

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if length(ARGS) < 2
    println(stderr, "Usage: julia scripts/assess.jl <status_N.h5> <database.h5>")
    exit(1)
end

status_file = ARGS[1]
database_file = ARGS[2]

m = match(r"status_(\d+)\.h5$", basename(status_file))
if m === nothing
    println(stderr, "Error: status file must be named status_N.h5")
    exit(1)
end
n = parse(Int, m.captures[1])
next_status_file = joinpath(dirname(status_file), "status_$(n+1).h5")

# ═══════════════════════════════════════════════════════════════════════════════
# Read inputs
# ═══════════════════════════════════════════════════════════════════════════════

trials = IO.read_trials(status_file)
misfits_map = IO.read_misfits(status_file)
strategy = IO.read_strategy(status_file)
config = IO.read_config(database_file)

xcorr_mat = get(misfits_map, :xcorr, nothing)
polarity_mat = get(misfits_map, :polarity, nothing)
psr_mat = get(misfits_map, :psr, nothing)
if xcorr_mat === nothing || polarity_mat === nothing || psr_mat === nothing
    println(stderr, "Error: /misfits must contain :xcorr, :polarity, :psr")
    exit(1)
end

N_phases = size(xcorr_mat, 1)
N_stations = size(polarity_mat, 1)
N_trials = length(trials.strike)

# ═══════════════════════════════════════════════════════════════════════════════
# Convert Int32 masks to Bool
# ═══════════════════════════════════════════════════════════════════════════════

xcorr_phase_mask = length(strategy.xcorr_phase_mask) >= N_phases ?
    Vector{Bool}(strategy.xcorr_phase_mask[1:N_phases] .== Int32(1)) :
    Vector{Bool}(trues(N_phases))
polarity_channel_mask = length(strategy.polarity_channel_mask) >= N_stations ?
    Vector{Bool}(strategy.polarity_channel_mask[1:N_stations] .== Int32(1)) :
    Vector{Bool}(trues(N_stations))
psr_channel_mask = length(strategy.psr_channel_mask) >= N_stations ?
    Vector{Bool}(strategy.psr_channel_mask[1:N_stations] .== Int32(1)) :
    Vector{Bool}(trues(N_stations))

# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate misfits
# ═══════════════════════════════════════════════════════════════════════════════

total, best_idx, per_module = aggregate_misfits(
    xcorr_mat, polarity_mat, psr_mat,
    xcorr_phase_mask, polarity_channel_mask, psr_channel_mask,
    strategy.module_weights,
)

best_sdr = [trials.strike[best_idx], trials.dip[best_idx], trials.rake[best_idx]]
best_depth_idx = trials.depth_idx[best_idx]
best_freq_idx = trials.freq_idx[best_idx]
best_misfit_val = total[best_idx]

# ═══════════════════════════════════════════════════════════════════════════════
# Per-depth misfits
# ═══════════════════════════════════════════════════════════════════════════════

depth_vals = get(config, "depth_vals", Float64[])
N_depths = length(depth_vals)
depth_misfits = fill(Inf, N_depths)
for j in 1:N_trials
    di = trials.depth_idx[j]
    if 1 <= di <= N_depths
        depth_misfits[di] = min(depth_misfits[di], total[j])
    end
end
depth_misfits[isinf.(depth_misfits)] .= NaN

# ═══════════════════════════════════════════════════════════════════════════════
# Per-frequency misfits
# ═══════════════════════════════════════════════════════════════════════════════

freq_bands_low = get(config, "freq_bands_low", Float64[])
N_frequencies = length(freq_bands_low)
freq_misfits = fill(Inf, N_frequencies)
for j in 1:N_trials
    fi = trials.freq_idx[j]
    if 1 <= fi <= N_frequencies
        freq_misfits[fi] = min(freq_misfits[fi], total[j])
    end
end
freq_misfits[isinf.(freq_misfits)] .= NaN

# ═══════════════════════════════════════════════════════════════════════════════
# Refine + prompt
# ═══════════════════════════════════════════════════════════════════════════════

trial_result = TrialResult(
    best_sdr, best_depth_idx, best_freq_idx, best_misfit_val,
    depth_misfits, freq_misfits,
)

next_strategy = refine_strategy(strategy, trial_result)
continue_flag = prompt_operator(best_sdr, best_misfit_val, strategy)

converged_val = continue_flag ? Int32(0) : Int32(1)
converged_reason = continue_flag ? "" : "user"

# Rebuild strategy with converged flag
final_strategy = IO.Strategy(
    next_strategy.strike0, next_strategy.dstrike, next_strategy.nstrike,
    next_strategy.dip0, next_strategy.ddip, next_strategy.ndip,
    next_strategy.rake0, next_strategy.drake, next_strategy.nrake,
    next_strategy.depth_indices,
    next_strategy.freq_indices,
    next_strategy.xcorr_phase_mask,
    next_strategy.polarity_channel_mask,
    next_strategy.psr_channel_mask,
    next_strategy.module_weights,
    next_strategy.best_sdr,
    next_strategy.best_depth_index,
    next_strategy.best_misfit,
    next_strategy.iteration,
    converged_val,
    converged_reason,
    next_strategy.freq_accumulated,
    next_strategy.freq_misfit_curve,
    next_strategy.depth_misfit_accumulated,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════════════════════════════

if continue_flag
    cp(status_file, next_status_file; force=true)
    IO.write_strategy(next_status_file, final_strategy)
    println("Wrote ", next_status_file, " (converged=", Int(final_strategy.converged), ")")
else
    IO.write_strategy(status_file, final_strategy)
    println("Set converged=1 on ", status_file)
end