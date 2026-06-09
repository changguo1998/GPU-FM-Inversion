# Design: CUDA-Accelerated Focal Mechanism Inversion

## Pipeline Architecture

A 4-stage pipeline. The driver script loops: setup → forward → assess, iterating on `status_{N}.h5` files until convergence.

```
setup → forward → assess → [setup → forward → assess → ...] → export
```

## Programs

| Program | Language | Role |
|---------|----------|------|
| `setup.jl` | Julia | First run: preprocess all data, write `database.h5` and `status_0.h5` (initial strategy + trials from config). Subsequent runs: read strategy from `status_{N}.h5`, generate trials into it. |
| `forward.cpp` | C++ / Kokkos | GPU misfit computation: for each trial, compute raw per-module misfit across all phases. No weighting or aggregation. |
| `assess.jl` | Julia | Apply module weights, aggregate misfits, check convergence, write `status_{N+1}.h5` with strategy for the next iteration. |
| `export.jl` | Julia | Compile final solution, write `output.h5`. |

## Data Files

| File | Lifetime | Contents |
|------|----------|----------|
| `raw.h5` | Static | External data only: event info, station metadata, raw SAC waveforms |
| `database.h5` | Static | All preprocessed data: Green's functions at all depths, all frequency-band-filtered waveform variants, per-module preprocessing output, algorithm configuration |
| `status_{N}.h5` | Per-iteration | Self-contained snapshot: strategy, trials, and misfits for iteration N |
| `output.h5` | Final | Best-fitting parameters, uncertainties, summary |

## Calling Chain

```
driver.sh
  │
  ├── setup.jl      reads:  raw.h5, database.h5/config
  │                  writes: database.h5 (once, first run only)
  │                  writes: status_0.h5 (/strategy + /trials)
  │
  ├── forward.cpp   reads:  database.h5, status_0.h5 (/trials)
  │                  writes: status_0.h5 (/misfits)
  │
  ├── assess.jl     reads:  status_0.h5 (/trials + /misfits)
  │                  writes: status_1.h5 (/strategy)
  │
  ├── setup.jl      reads:  database.h5, status_1.h5 (/strategy)
  │                  writes: status_1.h5 (+ /trials)
  │
  ├── forward.cpp   reads:  database.h5, status_1.h5 (/trials)
  │                  writes: status_1.h5 (+ /misfits)
  │
  ├── assess.jl     reads:  status_1.h5 (/trials + /misfits)
  │                  writes: status_2.h5 (/strategy)  — or signals done
  │
  ├── ... loop until assess.jl signals convergence ...
  │
  └── export.jl     reads:  last status_{N}.h5, database.h5
                     writes: output.h5
```

### raw.h5 — External data, static

Unprocessed input data from external sources. Never modified.

- Event info: origin time, magnitude, location
- Station metadata: location, azimuth, distance, channel inventory
- Raw SAC waveforms: observed traces per station × channel

### database.h5 — Preprocessed data, static

Written once by `setup.jl` on first run. Contains everything derived from `raw.h5` that `forward.cpp` needs:

- Pre-computed Green's functions: loaded from external program, stored at all required depths
- Preprocessed waveform variants: filtered copies of observed + Green's data for every frequency band in the search range
- Per-module preprocessing: PSR amplitude ratios, polarity picks — each module's derived data
- Algorithm configuration: misfit module list, parameter bounds, convergence criteria

### status_{N}.h5 — Per-iteration

One file per iteration. `status_0.h5` is created by `setup.jl` from config; subsequent files are started by `assess.jl` and completed by `setup.jl` + `forward.cpp`.

- **Strategy** (written by `assess.jl`; iteration 0 by `setup.jl`):
  - Search grid: SDR (`strike0, dstrike, nstrike` etc.), depth/freq index lists
  - Module weights (applied in assess, not forward)
  - Phase selection mask
  - Best result so far
- **Trial parameters** (written by `setup.jl`):
  - strike, dip, rake, depth for each trial
  - Reference indices (`freq_idx`, `depth_idx`) into `database.h5`
- **Raw misfits** (written by `forward.cpp`):
  - `misfit[N_modules × N_phases × N_trials]` — per-module, per-phase, per-trial

### output.h5 — Final

Written once by `export.jl`. Best-fitting source parameters with uncertainties, iteration history summary, station-level details.

## Principles

1. **`forward.cpp` is stateless.** Reads preprocessed data + trial parameters, outputs raw per-module misfits. No reduction, no weighting, no strategy knowledge.
2. **`assess.jl` owns the strategy.** Module weights, channel selection, convergence decisions — all happen post-forward.
3. **`setup.jl` precomputes everything upfront.** All frequency bands, all depth-dependent GF slices, all per-module preprocessing — done once, stored in static `database.h5`. Trials only index into precomputed slices.
4. **Per-iteration files.** Each `status_{N}.h5` is a self-contained snapshot of one iteration — inspectable and debuggable independently. No growing datasets, no cumulative state.
5. **HDF5 is the interface.** No shared memory, no RPC. Files are debuggable and inspectable at every step.

## GPU Computation (`forward.cpp`)

Built with Kokkos for portable GPU execution (CUDA, HIP, SYCL, OpenMP).

Single kernel: for each module × trial × phase × ... combination, compute misfit independently. No reduction or aggregation — `assess.jl` handles all weighting and summation.

Data volume for a typical event (~20 stations, ~120 phases, ~10k trials) is under 20 MB — fits entirely in GPU memory.