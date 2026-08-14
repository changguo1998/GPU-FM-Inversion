# Stage: `scripts/output.jl` — Output Compilation

## Role

Runs once after the assess loop converges. Reads `status_N.h5:/misfits` and
`/trials` plus `database.h5` metadata, selects the best trial from the XCorr
misfits, resolves it to physical values via `/paraspace`, and writes the final
`solution` / `uncertainty` / `per_phase` / `per_station_summary` / `summary`
groups to `output.h5`.

## Usage

```bash
DATA_DIR=<dir> julia scripts/output.jl
```

Files located via `ENV["DATA_DIR"]` (exported by driver.sh): `database.h5`,
`status/status_N.h5` (latest), writes `output.h5`.

## Best-trial selection

1. For each XCorr module (`XcorrP`/`XcorrS` present in `/misfits`), take the
   per-trial column mean of the `[entries × trials]` misfit matrix
   (NaN-safe).
1. Average those per-module means (equal weight — weighted aggregation is
   pending).
1. `best_idx = argmin(total)`. Physical values are resolved exclusively
   through `/paraspace` and the trial index vectors.

## Outputs (`output.h5`)

| Group | Contents |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `/solution` | strike, dip, rake, depth (deg/km), freq_idx, moment_tensor [6], misfit |
| `/uncertainty` | strike/dip/rake std over the best neighborhood (misfit ≤ 1.05×best), depth_range [min,max], freq_test_misfit_curve (NaN, unimplemented) |
| `/per_phase` | phase_id, station_id, phase_type, misfit_per_module [N_modules × N_phases], selected, cross_correlation (== intermediates cc_max at best trial) |
| `/per_station_summary` | station_id, n_phases, mean_cross_correlation, misfit_total (zeros, pending) |
| `/summary` | total_iterations, total_trials, convergence_reason |

## Simplifications / TODO (see `doc/roadmap.md` Phase 2)

- Best trial is driven by the XCorr misfit only; per-module weights and the
  cross-module aggregation belong to the pending assess weighting work.
- `freq_test_misfit_curve` is `NaN` (not implemented).
- `convergence_reason` is fixed to `"single iteration (refinement pending)"`.
- `per_station_summary.misfit_total` is zero-filled (pending).

## Notes

- HDF5.jl reads the C++ C-order `cc_max` as `[N_trials × N_phases]`; the
  per-phase cross-correlation column is taken as the best-trial row.
- Baseline (2026-08-14): best = true source (30, 60, 90) @ 10 km,
  misfit ≈ 0.1602 — see `AGENTS.md`.
