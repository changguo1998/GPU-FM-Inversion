# Stage: `scripts/output.jl` — Final Solution Compilation

## Role

Reads all files produced by prior pipeline stages and compiles the complete focal mechanism solution into `output.h5`. Uses `shared/mt/` for SDR→MT conversion and `shared/aggregate/` for misfit re-aggregation. Rearranges, summarizes, and gathers — performs no novel misfit computation.

**Exceptions** (novel computation):
- **Waveform synthesis** (`GF × MT`): generates synthetic seismograms for QC
- **Depth range**: applies 5% threshold to per-depth misfit accumulated by `assess.jl`

This stage was renamed from `export.jl`.

## Inputs

| Source             | Description                                                                                             |
|--------------------|---------------------------------------------------------------------------------------------------------|
| `status_{0..N}.h5` | Reads `/trials`, `/misfits`, `/strategy` from all completed status files (including the converged file) |
| `database.h5`      | Reads `/index`, `/greens`, `/data`, `/config`                                                           |

## Outputs

| Source      | Description                                                                                                         |
|-------------|---------------------------------------------------------------------------------------------------------------------|
| `output.h5` | Best-fit parameters, uncertainties, per-phase breakdown, per-station summary, optional synthetic waveforms, summary |

## Responsibilities

1. **Recompute best fit**: independently re-aggregate misfits to verify best trial matches `/strategy/best_sdr`
2. **SDR → MT conversion**: compute final moment tensor
3. **Synthesize frequency uncertainty**: read accumulated freq results from strategy → std of SDR across bands
4. **Synthesize depth range**: apply 5% tolerance to `depth_misfit_accumulated` → depth bounds
5. **Per-phase breakdown**: extract per-module misfit at best trial for each phase
6. **Waveform synthesis** (optional): compute final synthetic seismograms (`GF × MT`) for QC.
7. **Write output**: compile all results into `output.h5`

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down. Solution compilation is inline (no separate `solution_comp.jl`). Uses `Aggregate.aggregate_misfits`, `Aggregate.compute_depth_range`, `Aggregate.compute_sdr_std` from `shared/aggregate/`, `MT.sdr_to_mt` from `shared/mt/`, and `IO.find_latest_status` for file discovery.

## Tool Stack

- Julia (`HDF5.jl`, `Statistics.jl`, `LinearAlgebra.jl`)
- `MT` (shared/mt, SDR → MT conversion)
- `Aggregate` (shared/aggregate, misfit re-aggregation for verification)

## What It Does NOT Do

- Does NOT compute misfits
- Does NOT run grid search
- Does NOT modify any input files