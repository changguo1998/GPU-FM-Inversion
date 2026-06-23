#!/usr/bin/env julia
#=
Preprocess stage entry point.
Reads strategy from status_{N}.h5, generates trials, writes /trials back.

Usage:
  julia scripts/preprocess.jl <status_N.h5> [database.h5]

  database.h5 is optional — if provided, depth_vals are read from /config/depth_vals.
  Otherwise depth indices are used as-is (1-based mapping).
=#

using HDF5

# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if length(ARGS) < 1
    println(stderr, "Usage: julia scripts/preprocess.jl <status_N.h5> [database.h5]")
    exit(1)
end

status_file = ARGS[1]
database_file = length(ARGS) >= 2 ? ARGS[2] : nothing

if !isfile(status_file)
    println(stderr, "Error: status file not found: $status_file")
    exit(1)
end

if database_file !== nothing && !isfile(database_file)
    println(stderr, "Error: database file not found: $database_file")
    exit(1)
end

# ─────────────────────────────────────────────────────────
# Include modules
# ─────────────────────────────────────────────────────────

SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "..", "shared", "io", "src", "IO.jl"))
using .IO

include(joinpath(SCRIPT_DIR, "..", "shared", "grid", "src", "Grid.jl"))
using .Grid

# ─────────────────────────────────────────────────────────
# Read strategy
# ─────────────────────────────────────────────────────────

strategy = IO.read_strategy(status_file)
println("Read strategy from $status_file (iteration $(strategy.iteration))")

# ─────────────────────────────────────────────────────────
# Extract GridStrategy from IO.Strategy
# ─────────────────────────────────────────────────────────

grid = Grid.GridStrategy(
    strategy.strike0,
    strategy.dstrike,
    strategy.nstrike,
    strategy.dip0,
    strategy.ddip,
    strategy.ndip,
    strategy.rake0,
    strategy.drake,
    strategy.nrake,
    strategy.depth_indices,
    strategy.freq_indices,
    strategy.best_depth_index,
)

# ─────────────────────────────────────────────────────────
# Get depth values (from database.h5 or fallback)
# ─────────────────────────────────────────────────────────

if database_file !== nothing && IO.h5exists(database_file, "config/depth_vals")
    depth_vals = h5open(database_file, "r") do f
        read(f["config/depth_vals"])
    end
    println("Read $(length(depth_vals)) depth values from database.h5")
else
    # Fallback: use 1:max_index as synthetic depth values
    max_idx = isempty(grid.depth_indices) ? grid.best_depth_index : maximum(grid.depth_indices)
    depth_vals = collect(Float64, 1:max(max_idx, 1))
    println("Using synthetic depth values [1:$max_idx] (no database.h5 provided)")
end

# ─────────────────────────────────────────────────────────
# Generate trials
# ─────────────────────────────────────────────────────────

trialgen_trials = Grid.generate_trials(grid, depth_vals)
nt = length(trialgen_trials.strike)
n_s = max(grid.nstrike, 1)
n_d = max(grid.ndip, 1)
n_r = max(grid.nrake, 1)
n_depth = max(length(grid.depth_indices), 1)
n_freq = max(length(grid.freq_indices), 1)
println("Generated $nt trials ($n_s strikes × $n_d dips × $n_r rakes × $n_depth depths × $n_freq freqs)")

# ─────────────────────────────────────────────────────────
# Convert Grid.TrialSet → IO.TrialSet and write
# ─────────────────────────────────────────────────────────

hdf5_trials = IO.TrialSet(
    trialgen_trials.strike,
    trialgen_trials.dip,
    trialgen_trials.rake,
    trialgen_trials.depth,
    trialgen_trials.depth_idx,
    trialgen_trials.freq_idx,
)

IO.write_trials(status_file, hdf5_trials)
println("Written /trials to $status_file ($nt trials)")