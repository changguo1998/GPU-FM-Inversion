"""
Trial generation from strategy parameters (grid expansion).
Cartesian product of varying axes: strike × dip × rake × depth × freq.

Used by preprocess.jl on each loop iteration.
"""
module TrialGen

export generate_trials, TrialSet, GridStrategy

# ─────────────────────────────────────────────────────────
# Types (minimal subset for trial generation)
# ─────────────────────────────────────────────────────────

"""
    GridStrategy

Grid parameters extracted from the full Strategy struct in HDF5IO.jl.
Contains only the fields needed for trial generation.
"""
struct GridStrategy
    strike0::Float64
    dstrike::Float64
    nstrike::Int32
    dip0::Float64
    ddip::Float64
    ndip::Int32
    rake0::Float64
    drake::Float64
    nrake::Int32
    depth_indices::Vector{Int32}
    freq_indices::Vector{Int32}
    best_depth_index::Int32
end

"""
    TrialSet

Output of trial generation. Column vectors of length N_trials.
"""
struct TrialSet
    strike::Vector{Float64}
    dip::Vector{Float64}
    rake::Vector{Float64}
    depth::Vector{Float64}
    depth_idx::Vector{Int32}
    freq_idx::Vector{Int32}
end

# ─────────────────────────────────────────────────────────
# Axis expansion helper
# ─────────────────────────────────────────────────────────

"""
    expand_axis(var0, dvar, n) -> Vector{Float64}

Expand a single grid axis. Returns `n` values `var0 + i * dvar` for `i = 0:n-1`.
If `n <= 0`, the axis does not vary and returns `[var0]`.
"""
function expand_axis(var0::Float64, dvar::Float64, n::Int32)::Vector{Float64}
    if n <= 0
        return [var0]
    end
    return [var0 + Float64(i) * dvar for i in 0:(n-1)]
end

# ─────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────

"""
    generate_trials(strategy::GridStrategy, depth_vals::Vector{Float64}) -> TrialSet

Generate trials as the Cartesian product of varying axes:
strike (outermost) → dip → rake → depth → freq (innermost).

# Arguments
- `strategy::GridStrategy`: grid parameters from status file
- `depth_vals::Vector{Float64}`: actual depth values from database config, indexed by depth_idx

# Returns
- `TrialSet` with all trial vectors of length `N = nstrike × ndip × nrake × N_depths × N_freqs`

# Grid expansion rules
- Axis with `n <= 0` → not varying, single value `var0`
- Empty `depth_indices` → uses `best_depth_index` from strategy
- Empty `freq_indices` → no frequency variation, uses `Int32(1)`
- Trial order is deterministic: strike outermost, freq innermost
"""
function generate_trials(strategy::GridStrategy, depth_vals::Vector{Float64})::TrialSet
    # Expand SDR axes
    strikes = expand_axis(strategy.strike0, strategy.dstrike, strategy.nstrike)
    dips = expand_axis(strategy.dip0, strategy.ddip, strategy.ndip)
    rakes = expand_axis(strategy.rake0, strategy.drake, strategy.nrake)

    # Depth axis: empty → use best_depth_index
    if isempty(strategy.depth_indices)
        depth_idxs = Int32[strategy.best_depth_index]
    else
        depth_idxs = strategy.depth_indices
    end

    # Freq axis: empty → no variation, single band
    if isempty(strategy.freq_indices)
        freq_idxs = Int32[1]
    else
        freq_idxs = strategy.freq_indices
    end

    n_strikes = length(strikes)
    n_dips = length(dips)
    n_rakes = length(rakes)
    n_depths = length(depth_idxs)
    n_freqs = length(freq_idxs)

    n_trials = n_strikes * n_dips * n_rakes * n_depths * n_freqs

    # Pre-allocate
    strikes_out = Vector{Float64}(undef, n_trials)
    dips_out    = Vector{Float64}(undef, n_trials)
    rakes_out   = Vector{Float64}(undef, n_trials)
    depths_out  = Vector{Float64}(undef, n_trials)
    depth_idx_out = Vector{Int32}(undef, n_trials)
    freq_idx_out  = Vector{Int32}(undef, n_trials)

    idx = 1
    # strike outermost → dip → rake → depth → freq innermost
    for s in strikes
        for d in dips
            for r in rakes
                for didx in depth_idxs
                    # Lookup actual depth value with bounds check
                    if 1 <= didx <= length(depth_vals)
                        depth_val = depth_vals[didx]
                    else
                        depth_val = NaN
                    end
                    for fidx in freq_idxs
                        strikes_out[idx]   = s
                        dips_out[idx]      = d
                        rakes_out[idx]     = r
                        depth_idx_out[idx] = didx
                        depths_out[idx]    = depth_val
                        freq_idx_out[idx]  = fidx
                        idx += 1
                    end
                end
            end
        end
    end

    return TrialSet(strikes_out, dips_out, rakes_out, depths_out, depth_idx_out, freq_idx_out)
end

end # module