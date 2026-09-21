# AGENTS.md — Aggregate module (`shared/aggregate/`)

## Role

Misfit aggregation over the Operator × Phase × Output decomposition
(see `doc/misfit-decomposition.md`). Level 1 extractors transform raw
kernel intermediates into base misfit matrices; Level 2 composers aggregate
base misfit matrices into composed misfits (e.g. RelShift = StdDev of
AbsShiftP/S per station). Pure Julia — no HDF5 I/O, no waveform access.

Used by: `scripts/assess.jl`.

## Files

| File | Role |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| `src/Aggregate.jl` | Package entry and exports |
| `src/extractors.jl` | `EXTRACTORS` registry — keyed by `(operator, output)`, maps intermediates → Level 1 misfit matrices |
| `src/composers.jl` | `COMPOSERS` registry — keyed by aggregate operator, aggregates base misfits → Level 2 misfit matrices |
| `src/objective_primitives.jl` | Legacy PSR and normalized-polarity matrix evaluators；DSL 目标改由 `Misfit.evaluate_pipeline` 通用求值 |
| `src/hierarchical_sum.jl` | Channel → station → trial summation, with station-native objective support |
| `src/StdDev.jl` | StdDev operator — `RELATIVE_OFFSET`/`MEAN` outputs, per-station std/mean across base misfits |

## Extractors (`EXTRACTORS[(operator, output)]`)

| Key | Transform |
|-------------------------------|--------------------------------------------------|
| `(:Xcorr, :cc_max)` | `1 .- cc_max` (normalized CC misfit) |
| `(:Xcorr, :best_lag)` | `best_lag * dt` (absolute time shift in seconds) |
| PSR evaluator | squared natural-log RMS S/P residual |
| normalized polarity evaluator | L1 difference of L2-normalized signed amplitudes |

## Composers (`COMPOSERS[operator]`)

| Key | Bases | Output |
|-----------|--------------------------|---------------------------------------------------------------|
| `:StdDev` | per-station base misfits | `:relative_offset` (std), `:mean` — `[N_stations × N_trials]` |

RelShift is registered in sample configs as `Aggregate.StdDev` with
`bases = [:AbsShiftP, :AbsShiftS]` and `output = RELATIVE_OFFSET`.

## Hierarchical sum

`hierarchical_sum` applies no averaging, normalization, or weights. It sums
phase-indexed values per physical channel, adds channel-indexed values, sums channels
by station, adds native station-indexed values, then sums stations into one value per
trial. Signed Lag modules pass through `absolute_modules` for aggregation only.

## Coding conventions

- 4-space indent, `Dict`-keyed registries, lambdas for transforms.
- Composer `ctx` carries `N_stations` and per-base `base_station_idx` row → station mapping.
- Level 2 instances have no instance module and no preprocessing (skipped by `input.jl`).
