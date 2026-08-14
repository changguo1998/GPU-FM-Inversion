# Trial Generation

"""default_grid() -> NamedTuple (see IO.DEFAULT_GRID)

Default 5° search grid over the full parameter space (single source of truth:
`H5IO.DEFAULT_GRID`): strike 0:5:355 (72), dip 0:5:90 (19), rake -90:5:90 (37).
"""
default_grid() = H5IO.DEFAULT_GRID



# Axis expansion helper

"""expand_axis(var0, dvar, n) -> Vector{Float64}; expands one grid axis to
`n` values `var0 + i*dvar` for i = 0:n-1 (`[var0]` when n <= 0)."""
function expand_axis(var0::Float64, dvar::Float64, n::Int32)::Vector{Float64}
    if n <= 0
        return [var0]
    end
    return [var0 + Float64(i) * dvar for i in 0:(n - 1)]
end

# Main function

"""generate_trials(strategy) -> H5IO.TrialSet

Cartesian product of varying axes — strike → dip → rake → depth → freq.
Depth/freq use the strategy's index subsets. Trials carry index-only fields
(1-based into `/paraspace`); physical values resolved on forward/output.
"""
function generate_trials(strategy::H5IO.Strategy)::H5IO.TrialSet
    # Axis lengths: SDR from grid dims (n<=0 => single point); depth/freq from indices
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
