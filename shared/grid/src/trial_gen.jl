# Trial Generation

"""
    default_grid() -> NamedTuple (see IO.DEFAULT_GRID)

Default initial search grid covering the full parameter space at 5° resolution
(single source of truth: `IO.DEFAULT_GRID`).
strike: 0:5:355 (72, full circle), dip: 0:5:90 (19), rake: -90:5:90 (37).
"""
default_grid() = H5IO.DEFAULT_GRID


# Note: uses `H5IO.Strategy` (full 12-field grid definition) directly and
# returns `H5IO.TrialSet` — no separate GridStrategy/Grid.TrialSet types.

# Axis expansion helper

"""
    expand_axis(var0, dvar, n) -> Vector{Float64}

Expand a single grid axis. Returns `n` values `var0 + i * dvar` for `i = 0:n-1`.
If `n <= 0`, the axis does not vary and returns `[var0]`.
"""
function expand_axis(var0::Float64, dvar::Float64, n::Int32)::Vector{Float64}
    if n <= 0
        return [var0]
    end
    return [var0 + Float64(i) * dvar for i in 0:(n - 1)]
end

# Main function

"""
    generate_trials(strategy::H5IO.Strategy) -> H5IO.TrialSet

Generate trials as the Cartesian product of varying axes:
strike (outermost) → dip → rake → depth → freq (innermost).
Depth/freq axes use the strategy's `depth_indices`/`freq_indices` subsets.
Trials carry **indices only** (strike_idx/dip_idx/rake_idx/depth_idx/freq_idx,
1-based into `/paraspace` axes) — physical values live exclusively in
`/paraspace` and are resolved on forward (MT) / output.
"""
function generate_trials(strategy::H5IO.Strategy)::H5IO.TrialSet
    # Axis lengths: SDR from grid dims (n<=0 => single point), depth/freq from indices
    strike_idxs = Int32.(1:max(Int(strategy.nstrike), 1))
    dip_idxs = Int32.(1:max(Int(strategy.ndip), 1))
    rake_idxs = Int32.(1:max(Int(strategy.nrake), 1))

    if isempty(strategy.depth_indices)
        depth_idxs = Int32[1]
    else
        depth_idxs = strategy.depth_indices
    end

    if isempty(strategy.freq_indices)
        freq_idxs = Int32[1]
    else
        freq_idxs = strategy.freq_indices
    end

    n_strikes = length(strike_idxs)
    n_dips = length(dip_idxs)
    n_rakes = length(rake_idxs)
    n_depths = length(depth_idxs)
    n_freqs = length(freq_idxs)

    n_trials = n_strikes * n_dips * n_rakes * n_depths * n_freqs

    strikes_out = Vector{Int32}(undef, n_trials)
    dips_out = Vector{Int32}(undef, n_trials)
    rakes_out = Vector{Int32}(undef, n_trials)
    depth_idx_out = Vector{Int32}(undef, n_trials)
    freq_idx_out = Vector{Int32}(undef, n_trials)

    idx = 1
    for s in strike_idxs
        for d in dip_idxs
            for r in rake_idxs
                for didx in depth_idxs
                    for fidx in freq_idxs
                        strikes_out[idx] = s
                        dips_out[idx] = d
                        rakes_out[idx] = r
                        depth_idx_out[idx] = didx
                        freq_idx_out[idx] = fidx
                        idx += 1
                    end
                end
            end
        end
    end

    return H5IO.TrialSet(strikes_out, dips_out, rakes_out, depth_idx_out, freq_idx_out)
end
