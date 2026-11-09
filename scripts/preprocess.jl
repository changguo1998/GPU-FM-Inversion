#!/usr/bin/env julia
#
# preprocess.jl - Trial generation.
#
# Reads /strategy from the latest status_N.h5, generates a budget-constrained
# Cartesian product of index values, and writes /trials for the forward stage.
#
# Usage:
#   DATA_DIR=<dir> julia scripts/preprocess.jl
#
# No CLI arguments; driver.sh supplies DATA_DIR and stores status_N.h5 in status/.

using HDF5

using StageLog

using IO, Search

# === 1. Locate files ===
data_dir = ENV["DATA_DIR"]
StageLog.setup_logger!("preprocess", joinpath(data_dir, "preprocess.log"))

status_dir = joinpath(data_dir, "status")

status_path, iter_n = IO.find_latest_status(status_dir)

@info "=" ^ 70
@info "preprocess stage started"
@info "  data dir     = $data_dir"
@info "  status file  = $status_path (iteration $iter_n)"

# === 2. Read the full-grid strategy ===
strategy = IO.read_strategy(status_path)

# === 3. Plan the budget and generate index-only trials ===
@info "  grid: strike $(strategy.nstrike) × dip $(strategy.ndip) × rake $(strategy.nrake) @ $(strategy.dstrike)°, depth $(length(strategy.depth_indices)), freq $(length(strategy.freq_indices)), duration $(length(strategy.duration_indices))"
full_trial_count = prod((
    max(Int(strategy.nstrike), 1),
    max(Int(strategy.ndip), 1),
    max(Int(strategy.nrake), 1),
    max(length(strategy.depth_indices), 1),
    max(length(strategy.freq_indices), 1),
    max(length(strategy.duration_indices), 1),
))
budget_text = get(ENV, "FM_TRIAL_BUDGET", "")
budget = if isempty(strip(budget_text))
    full_trial_count
else
    try
        parse(Int, strip(budget_text))
    catch
        error("FM_TRIAL_BUDGET must be a positive integer, got '$budget_text'")
    end
end
budget > 0 || error("FM_TRIAL_BUDGET must be a positive integer, got '$budget_text'")
plan = Search.budgeted_plan(strategy, budget)
budget_label = budget == full_trial_count ? "full ($full_trial_count)" : string(budget)
@info "  trial budget: $budget_label"
@info "  selected axes: strike $(length(plan.strike_indices)), dip $(length(plan.dip_indices)), rake $(length(plan.rake_indices)), depth $(length(plan.depth_indices)), freq $(length(plan.freq_indices)), duration $(length(plan.duration_indices))"
t0 = time()
trials = Search.generate_trials(plan)
elapsed = time() - t0

@info "  generated $(length(trials.strike_idx)) trials in $(round(elapsed, digits = 3)) s"

# === 5. Write /trials ===
IO.write_trials(status_path, trials)
@info "  wrote /trials to $status_path"
