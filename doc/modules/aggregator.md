# Module: Misfit Aggregator

## Purpose

Apply per-module masks, weight, and aggregate raw misfits into per-trial total scores. Used by assess.jl (primary) and export.jl (verification).

## Used By

- `assess.jl` — per-iteration aggregation
- `export.jl` — independent verification of best trial

## Input

| Source | Shape | Level |
|--------|-------|-------|
| `xcorr` | `[N_phases × N_trials]` | phase |
| `polarity` | `[N_stations × N_trials]` | station P-polarity |
| `psr` | `[N_stations × N_trials]` | station |
| `xcorr_phase_mask` | `[N_phases]` | XCorr mask |
| `polarity_station_mask` | `[N_stations]` | Polarity mask |
| `psr_station_mask` | `[N_stations]` | PSR mask |
| `module_weights` | `[N_modules]` | scalar per module |

## Processing Steps

1. **Per-module masking**: set masked phases/stations to NaN using the module-specific mask for each input shape
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

## Testing Strategy

- Verify aggregation matches hand-computed examples
- NaN handling: masked phases don't affect trial totals
- Weight=0: module contributes nothing
- Cross-validate: assess.jl and export.jl produce same best_idx
