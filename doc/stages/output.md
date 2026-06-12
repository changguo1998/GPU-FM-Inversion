# Stage: `output.jl` — Final Solution Compilation

## Role

Reads all files produced by prior pipeline stages and compiles the complete focal mechanism solution into `output.h5`. Rearranges, summarizes, and gathers — performs no novel misfit computation.

**Exceptions** (novel computation):
- **Waveform synthesis** (`GF × MT`): generates synthetic seismograms for QC
- **Depth range**: applies 5% threshold to per-depth misfit accumulated by `assess.jl`

This stage was renamed from `export.jl`.

## Inputs

| Source | Description |
|--------|-------------|
| `status_{0..N}.h5` | Reads `/trials`, `/misfits`, `/strategy` (final + history) |
| `database.h5` | Reads `/index`, `/greens`, `/data`, `/config` |

## Outputs

| Source | Description |
|--------|-------------|
| `output.h5` | Best-fit parameters, uncertainties, per-station breakdown, synthetic waveforms, summary |

## Responsibilities

1. **Recompute best fit**: independently re-aggregate misfits to verify best trial matches `/strategy/best_sdr`
2. **SDR → MT conversion**: compute final moment tensor
3. **Synthesize frequency uncertainty**: read accumulated freq results from strategy → std of SDR across bands
4. **Synthesize depth range**: apply 5% tolerance to `depth_misfit_accumulated` → depth bounds
5. **Per-station breakdown**: extract per-module misfit at best trial
6. **Waveform synthesis** (optional): compute final synthetic seismograms (`GF × MT`) for QC
7. **Write output**: compile all results into `output.h5`

## Tool Stack

- Julia (`HDF5.jl`, `Statistics.jl`, `LinearAlgebra.jl`)
- `AssessUtils` (shared aggregator from assess stage, for verification)
- `MTUtils` (shared SDR → MT conversion)

## What It Does NOT Do

- Does NOT compute misfits
- Does NOT run grid search
- Does NOT modify any input files