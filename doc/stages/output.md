# Stage: `scripts/output.jl` — Output Compilation

## Role

Runs once after the assess loop converges. Reads `status_N.h5:/misfits` and
`/trials` plus `database.h5` metadata, selects the best trial from the XCorr
misfits, resolves it to physical values via `/paraspace`, and writes the final
`solution` / `uncertainty` / `per_phase` / `per_station_summary` / `summary`
groups to `output.h5` and a compact machine-readable `result.toml` report.

## Usage

```bash
DATA_DIR=<dir> julia scripts/output.jl
```

Files located via `ENV["DATA_DIR"]` (exported by driver.sh): `database.h5`,
`status/status_N.h5` (latest), writes `output.h5` and `result.toml`.

## Best-trial selection

1. Read `/aggregate/total`, produced by assess through channel → station → trial sums.
1. `best_idx = argmin(total)`. Physical values are resolved exclusively
   through `/paraspace` and the trial index vectors.

## Outputs (`output.h5`)

| Group | Contents |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `/solution` | strike, dip, rake, depth, duration (Gaussian σ in s), freq_idx/duration_idx, moment_tensor [6], misfit |
| `/uncertainty` | strike/dip/rake std over the best neighborhood (misfit ≤ 1.05×best), depth_range [min,max], freq_test_misfit_curve (NaN, unimplemented) |
| `/per_phase` | phase_id, station_id, phase_type, misfit_per_module [N_modules × N_phases], selected, cross_correlation (== intermediates cc_max at best trial) |
| `/per_station_summary` | station_id, n_phases, mean_cross_correlation, misfit_per_module, hierarchical misfit_total |
| `/summary` | total_iterations, total_trials, convergence_reason |

## Text output (`result.toml`)

The same compact result is written as standard TOML text at
`DATA_DIR/result.toml`. It contains `solution`, `uncertainty`, `summary`,
`per_phase`, and `per_station_summary`. The phase misfit matrix is serialized
as an array of row arrays, with its module names in `misfit_modules`.
Waveforms, Green's functions, and other large intermediate arrays are omitted.

`scripts/report.jl` consumes this file and writes the human-readable
`DATA_DIR/report.md`; see `doc/stages/report.md`.

## Simplifications / TODO (see `doc/roadmap.md`)

- `freq_test_misfit_curve` is `NaN` (not implemented).
- `convergence_reason` is fixed to `"single iteration"`.
- `per_station_summary.misfit_per_module` preserves summed module contributions at the best trial.

## Notes

- HDF5.jl reads the C++ C-order `cc_max` as `[N_trials × N_phases]`; the
  per-phase cross-correlation column is taken as the best-trial row.
- Baseline (2026-09-21): best = (210, 30, 90) @ 10 km, σ=0.2 s,
  misfit ≈ 2.06561e-3；该解与真值 (30, 60, 90) 的 moment tensor 相同。
