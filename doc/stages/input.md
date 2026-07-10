# Stage: `scripts/input.jl` — Data Ingestion & Initialization

## Role

Runs once at the start of the pipeline (before the main loop). Reads `config.jl`, locates external data (waveforms, station metadata, phase picks, Green's functions) via `Config.load_*()` interface, preprocesses all data into `database.h5` using `shared/signal/` for filtering/trimming and `shared/io/` for HDF5 I/O, and writes the initial strategy into `status_0.h5`.

当前为从头开发的第一阶段。输出 `database.h5` 和 `status_0.h5` 作为后续阶段的接口契约。

## Inputs

| Source | Description |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config.jl` | Bootstrap configuration: frequency bands, depth range, initial grid params, module settings, paths to external data (waveforms, station metadata, Green's functions) |

## Outputs

| Source | Description |
|---------------|--------------------------------------------------------------------------------------------------------------------------------|
| `database.h5` | All preprocessed data: Green's functions at all depths, filtered waveform variants, per-module preprocessing, algorithm config |
`status_0.h5` | Initial strategy (`/strategy` group) — search grid from `config.jl`. No trials yet.

## Responsibilities

1. **Preprocess raw data**: filter waveforms to frequency bands, trim time windows, extract XCorr and Polarity preprocessing output (GF preprocessed independently per depth), store in `database.h5`
1. **Load Green's functions**: read external GF files, store by phase × depth in `database.h5`
1. **Write algorithm config**: load `config.jl`, write into `database.h5`
1. **Write initial strategy**: initial search grid from config → `/strategy` in `status_0.h5`
1. **Write phase index**: build `/index` group in `database.h5` — phase IDs, types, station indices, distances, azimuths, and GF depth index mapping
1. **Create file skeleton**: `status_0.h5` is created with `/strategy` populated.
1. **Per-depth GF preprocessing**: each trial depth independently filters and windows its own Green's functions during XCorr/Polarity preprocessing (previously all depths reused the first depth's GF).

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down when `include`d or executed.

Tooling functions (time parsing, distance/azimuth computation, phase ID extraction) live in `shared/io/` (module `IO`) and are called as `IO.parse_time_iso`, `IO.haversine_distance`, etc.

- Julia (`HDF5.jl`, `DSP.jl` via `shared/signal/`, `Dates.jl`). PSR preprocessing not called by current input.jl (no PSR data stored in database.h5).
- Butterworth bandpass filter (DSP.jl, zero-phase forward-backward)
- Time-window trimming
- Green's function loader

## What It Does NOT Do

- Does NOT generate trials (future `preprocess.jl`)
- Does NOT compute misfits (future forward stage)
- Does NOT apply weights or make strategy decisions (future `assess.jl`)
- Does NOT run more than once per pipeline invocation
