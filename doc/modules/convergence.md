# Module: Grid Refinement & Strategy Update

## Purpose

Compute the next iteration's search grid from current results after each pipeline loop. The pipeline pauses after each iteration — the operator decides whether to continue or stop.

## Used By

- `assess.jl`

## Pause / Continue

`assess.jl` prompts the operator after writing refined results:

```
Best SDR: (strike=45.0, dip=30.0, rake=90.0), Misfit=0.032
Current grid: strike=45.0±10.0°, dip=30.0±8.0°, rake=90.0±8.0°
Continue? [y/N]
```

- **y** → writes `status_{N+1}.h5` with `/strategy/converged=0`. Driver loops to preprocess for next iteration.
- **N** (any other) → writes `status_{N+1}.h5` with `/strategy/converged=1`, `convergence_reason="user"`. Driver detects converged=1 and breaks to output.

## Grid Refinement

Computes next iteration's grid parameters from current best trial:

| Parameter | Source | Rule |
|-----------|--------|------|
| `strike0`, `dip0`, `rake0` | Current best trial SDR | New grid center |
| `dstrike`, `ddip`, `drake` | Current step sizes | Halved: `new_step = current_step / 2` |
| `nstrike`, `ndip`, `nrake` | Fixed | Always `[3, 3, 3]` (3×3×3 SDR grid around best) |
| `depth_indices` | `depth_misfit_accumulated` | Indices of depths within 20% of best depth misfit |
| `freq_indices` | `freq_misfit_curve` | Indices of frequencies within 20% of best frequency misfit |

**Refinement factor**: fixed at 0.5 (half step sizes each iteration).

**Grid size**: fixed at 3 per SDR axis. Total trials = `3 × 3 × 3 × N_depths × N_freqs`.

### Edge Cases

- Depth subset empty → use best depth index only (`N_depths = 1`)
- Frequency subset empty → use best freq index only (`N_freqs = 1`)
- First iteration (`status_0.h5`) → initial strategy set by config, no refinement

## Output

Updated strategy for `status_{N+1}.h5`:
- `strike0`, `dstrike`, `nstrike`
- `dip0`, `ddip`, `ndip`
- `rake0`, `drake`, `nrake`
- `depth_indices`
- `freq_indices`
- `best_sdr`: `[strike, dip, rake]` from current best trial
- `best_misfit`: total misfit of current best trial
- `freq_accumulated`: best SDR per frequency band, for uncertainty reporting
- `freq_misfit_curve`: misfit values used for frequency subset selection
- `depth_misfit_accumulated`: best misfit per depth, for depth subset selection

## Testing Strategy

- Grid center matches best trial from current iteration
- Step sizes halved correctly
- Grid size: always 3×3×3 SDR
- Depth/freq subsets contain best indices
