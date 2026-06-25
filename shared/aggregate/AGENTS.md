# AGENTS.md — Aggregate module (`shared/aggregate/src/Aggregate.jl`)

## Role

Misfit aggregation: apply per-module masks and weights to raw misfit matrices, produce per-trial total scores, find best trial. Also provides uncertainty computation helpers.

Used by: `assess.jl`, `output.jl`.

## Dependencies

- `Statistics` — `std` for frequency-uncertainty computation

## Exports

### Main function

`aggregate_misfits(xcorr, polarity, psr, xcorr_phase_mask, polarity_channel_mask, psr_channel_mask, module_weights)`

```
Returns: (total::Vector{Float64}[N_trials], best_idx::Int, per_module::Dict)
```

**Shapes:**

- `xcorr`: `[N_phases × N_trials]`
- `polarity`: `[N_channels × N_trials]`
- `psr`: `[N_channels × N_trials]`
- Masks: `Vector{Bool}` matching first dimension of each matrix
- `module_weights`: `[2]` or `[3]` = `[w_xc, w_pol, (w_psr)]`

**Algorithm:**

1. Per-module masked sum — for each trial, sum non-NaN values where mask is `true`. If no values contribute (all NaN or all masked), trial score = NaN for that module.
1. All-NaN check across all modules — throws `ErrorException` if every trial is NaN everywhere.
1. Weighted combination — NaN contributions are skipped (treated as 0.0). Module with weight=0 contributes nothing.
1. Best trial = minimum total (ignoring NaN-only trials).

**NaN handling:**

- Masked entries (mask=false) skipped entirely
- NaN values in active rows skipped per-row (do not propagate to trial total)
- Trial with zero contribution across all active rows per module → that module's score = NaN
- NaN module scores treated as 0 in weighted sum (module effectively contributes nothing)

### Uncertainty helpers

| Function              | Signature                                        | Returns                                                                |
|-----------------------|--------------------------------------------------|------------------------------------------------------------------------|
| `compute_depth_range` | `(depth_vals, depth_misfit_vec; tolerance=0.05)` | `[min_depth, max_depth]` or `[NaN, NaN]`                               |
| `compute_sdr_std`     | `(freq_accumulated[N_freq×3])`                   | `(strike_std, dip_std, rake_std)` or `(NaN, NaN, NaN)` if ≤1 frequency |

`compute_depth_range`: depths within `tolerance` fraction of best misfit. `compute_sdr_std`: standard deviation of best SDR per frequency band.

## Coding conventions

- All inputs are `Matrix{Float64}`, masks `Vector{Bool}`, weights `Vector{Float64}`.
- Accepts 2 or 3 module weights. When only 2 weights, PSR module is dropped from output.
- No HDF5 I/O, no pipeline state — pure computation.
