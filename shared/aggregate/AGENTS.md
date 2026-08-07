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
|---------------------|-------------------------------------------------------------------------------------------------------|
| `src/Aggregate.jl` | Package entry, includes StdDev/extractors/composers, exports `EXTRACTORS`, `COMPOSERS` |
| `src/extractors.jl` | `EXTRACTORS` registry — keyed by `(operator, output)`, maps intermediates → Level 1 misfit matrices |
| `src/composers.jl` | `COMPOSERS` registry — keyed by aggregate operator, aggregates base misfits → Level 2 misfit matrices |
| `src/StdDev.jl` | StdDev operator — `RELATIVE_OFFSET`/`MEAN` outputs, per-station std/mean across base misfits |

## Extractors (`EXTRACTORS[(operator, output)]`)

| Key | Transform |
|---------------------------|--------------------------------------------------------|
| `(:Xcorr, :cc_max)` | `1 .- cc_max` (normalized CC misfit) |
| `(:Xcorr, :best_lag)` | `best_lag * dt` (absolute time shift in seconds) |
| `(:Polarity, :syn_sign)` | mismatch vs observed polarity (`syn_sign .!= obs_pol`) |
| `(:Polarity, :dot_value)` | `abs(dot_value)` (confidence weight) |

## Composers (`COMPOSERS[operator]`)

| Key | Bases | Output |
|-----------|--------------------------|---------------------------------------------------------------|
| `:StdDev` | per-station base misfits | `:relative_offset` (std), `:mean` — `[N_stations × N_trials]` |

RelShift is registered in sample configs as `Aggregate.StdDev` with
`bases = [:AbsShiftP, :AbsShiftS]` and `output = RELATIVE_OFFSET`.

## Coding conventions

- 4-space indent, `Dict`-keyed registries, lambdas for transforms.
- Composer `ctx` carries `N_stations` and per-base `base_station_idx` row → station mapping.
- Level 2 instances have no instance module and no preprocessing (skipped by `input.jl`).
