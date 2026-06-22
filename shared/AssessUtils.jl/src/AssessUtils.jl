module AssessUtils

export aggregate_misfits

"""
    aggregate_misfits(
        xcorr, polarity, psr,
        xcorr_phase_mask, polarity_channel_mask, psr_channel_mask,
        module_weights,
    ) -> (total::Vector{Float64}, best_idx::Int, per_module::Dict{Symbol, Vector{Float64}})

Apply per-module masks, weight, and aggregate raw misfits into per-trial total scores.

# Arguments
- `xcorr`: shape `[N_phases × N_trials]` — raw XCorr misfits
- `polarity`: shape `[N_channels × N_trials]` — raw Polarity misfits
- `psr`: shape `[N_channels × N_trials]` — raw PSR misfits
- `xcorr_phase_mask`: length `N_phases` — `true` = active, `false` = masked (skip)
- `polarity_channel_mask`: length `N_channels` — `true` = active, `false` = masked
- `psr_channel_mask`: length `N_channels` — `true` = active, `false` = masked
- `module_weights`: `[3]` — weights for `[xcorr, polarity, psr]`

# Returns
- `total`: `[N_trials]` — weighted misfit per trial
- `best_idx`: index of trial with minimum total misfit (1-based)
- `per_module`: Dict with keys `:xcorr`, `:polarity`, `:psr` → `[N_trials]`

# NaN Handling
- Masked entries (where mask is `false`) are skipped entirely — their NaN values
  do not propagate to the trial total.
- If a trial has ALL entries NaN across ALL modules, an `ErrorException` is thrown.
- A module with weight=0 contributes nothing.

# Example
```julia
xcorr = [1.0 2.0; 3.0 4.0]           # 2 phases × 2 trials
polarity = [0.0 1.0]                   # 1 station × 2 trials
psr = [0.5 0.5]                        # 1 station × 2 trials
mask_xc = [true, true]                 # both phases active
mask_pol = [true]                      # station active
mask_psr = [true]                      # station active
weights = [1.0, 1.0, 1.0]             # equal weights

total, best, per_mod = aggregate_misfits(xcorr, polarity, psr,
    mask_xc, mask_pol, mask_psr, weights)
# total ≈ [4.5, 7.5], best = 1
```
"""
function aggregate_misfits(
    xcorr::Matrix{Float64},
    polarity::Matrix{Float64},
    psr::Matrix{Float64},
    xcorr_phase_mask::Vector{Bool},
    polarity_channel_mask::Vector{Bool},
    psr_channel_mask::Vector{Bool},
    module_weights::Vector{Float64},
)
    n_trials = size(xcorr, 2)

    # ── Per-module masked sum ──
    # For each trial: sum values where mask is true AND value is not NaN.
    function _masked_sum(data::Matrix{Float64}, mask::Vector{Bool})
        n_rows = size(data, 1)
        scores = zeros(Float64, n_trials)
        row_used = falses(n_trials)  # track if ANY non-NaN value contributed
        for i in 1:n_rows
            if !mask[i]
                continue  # masked row: skip entirely
            end
            for j in 1:n_trials
                v = data[i, j]
                if !isnan(v)
                    scores[j] += v
                    row_used[j] = true
                end
            end
        end
        # Trials that had no contribution → NaN
        for j in 1:n_trials
            if !row_used[j]
                scores[j] = NaN
            end
        end
        return scores
    end

    xc_per_trial = _masked_sum(xcorr, xcorr_phase_mask)
    pol_per_trial = _masked_sum(polarity, polarity_channel_mask)
    psr_per_trial = _masked_sum(psr, psr_channel_mask)

    # ── All-NaN check across all modules ──
    all_nan_xc = all(isnan, xc_per_trial)
    all_nan_pol = all(isnan, pol_per_trial)
    all_nan_psr = all(isnan, psr_per_trial)
    if all_nan_xc && all_nan_pol && all_nan_psr
        error("aggregate_misfits: all trials are NaN across all modules — check input data and masks")
    end

    # ── Apply module weights ──
    w_xc, w_pol, w_psr = module_weights[1], module_weights[2], module_weights[3]

    # ── Combine: weighted sum, treating NaN scores as 0 contribution ──
    function _add_weighted(a, w, b)
        for j in 1:n_trials
            val = a[j]
            if !isnan(val)
                b[j] += w * val
            end
        end
    end

    total = zeros(Float64, n_trials)
    _add_weighted(xc_per_trial, w_xc, total)
    _add_weighted(pol_per_trial, w_pol, total)
    _add_weighted(psr_per_trial, w_psr, total)

    # Edge case: if all weights are 0, total is all zeros — find any valid trial
    if all(t -> t == 0.0, total) && all(module_weights .== 0.0)
        best_idx = 1
    else
        # Find best trial (minimum total), ignoring NaN-only trials
        best_idx = 0
        best_val = Inf
        for j in 1:n_trials
            if total[j] < best_val
                best_val = total[j]
                best_idx = j
            end
        end
        if best_idx == 0
            error("aggregate_misfits: no valid trial found")
        end
    end

    per_module = Dict{Symbol, Vector{Float64}}(
        :xcorr => xc_per_trial,
        :polarity => pol_per_trial,
        :psr => psr_per_trial,
    )

    return (total, best_idx, per_module)
end

end # module