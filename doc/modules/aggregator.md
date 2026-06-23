# Module: Aggregate (Misfit Aggregation)

**Location**: `shared/aggregate/` (Julia package `Aggregate`)

## Purpose

Apply per-module masks, weight, and aggregate raw misfits into per-trial total scores. Used by assess.jl (primary) and output.jl (verification).

## Used By

- `assess.jl` — per-iteration aggregation
- `output.jl` — independent verification of best trial

## Input

| Source | Shape | Level |
|--------|-------|-------|
| `xcorr` | `[N_phases × N_trials]` | phase |
| `polarity` | `[N_channels × N_trials]` | channel P-polarity |
| `psr` | `[N_channels × N_trials]` | channel P/S ratio |
| `xcorr_phase_mask` | `[N_phases]` | XCorr mask |
| `polarity_channel_mask` | `[N_channels]` | Polarity mask |
| `psr_channel_mask` | `[N_channels]` | PSR mask |
| `module_weights` | `[N_modules]` | scalar per module |

## Processing Steps

1. **Per-module masking**: set masked phases/channels to NaN using the module-specific mask for each input shape
2. **Per-module aggregation**: sum valid misfits → per-trial score per module
3. **Apply module weight**: multiply by `module_weights[m]`
4. **Combine modules**: sum weighted scores → `[N_trials]`

## NaN Handling

- Masked or missing data → NaN
- NaN values skipped in sums (do NOT propagate NaN to entire trial)
- Error only if ALL trials for ALL modules are NaN (data/config problem)

## Output

| Output | Shape | Description |
|--------|-------|-------------|
| `total` | `[N_trials]` | Weighted misfit per trial |
| `best_idx` | scalar | Index of trial with minimum total misfit |

## Uncertainty Helpers

Two additional exported functions moved from `scripts/output.jl` during the flatten refactor:

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `compute_depth_range` | `depth_vals, depth_misfit_vec; tolerance=0.05` | `[min, max]` or `[NaN, NaN]` | Depths within tolerance fraction of best |
| `compute_sdr_std` | `freq_accumulated [N_freq × 3]` | `(s_std, d_std, r_std)` | SDR std across frequency bands |

## Testing Strategy

- Verify aggregation matches hand-computed examples
- NaN handling: masked phases don't affect trial totals
- Weight=0: module contributes nothing
- Cross-validate: assess.jl and output.jl produce same best_idx
