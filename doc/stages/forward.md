# Stage: `forward.cpp` — GPU Misfit Computation

## Role

GPU-accelerated core. Stateless misfit computation: reads preprocessed data + trial params, computes raw per-module misfits. No weighting, no aggregation, no strategy knowledge.

## Inputs

| Source | Description |
|--------|-------------|
| `database.h5` | Preprocessed data (read-only): waveform variants, index, Greens |
| `status_{N}.h5` | Trials (`/trials` group): parameters + data slice references |

## Outputs

| Source | Description |
|--------|-------------|
| `status_{N}.h5` | Misfits (`/misfits/` group): one dataset per module, raw (unweighted) |

## Responsibilities

1. **Read inputs**: load trials and preprocessed data from HDF5
2. **Transfer to GPU**: move all needed data to GPU memory once
3. **Precompute on GPU**: module-specific reduction (e.g., time-domain CC for XCorr, summed GF for Polarity, amplitude covariances for PSR)
4. **Launch kernels**: for each enabled module, launch misfit kernel (trial × phase grid)
5. **Write results**: copy misfits back to host, write to `status_{N}.h5`

## Execution Model

- Load all data once to GPU → precompute → launch kernels back-to-back → write
- No data movement between modules (all reduction data fits in GPU memory)
- Linear decomposition: precompute `CC(obs, GF[:,i])` per phase; per-trial: weighted sum of precomputed CCs

## Tool Stack

- C++17 + Kokkos (portable GPU parallelism)
- HDF5 C API (no HighFive)
- SDR → MT conversion (shared with Julia via `mt_convert.jl` / `MTUtils`)

## Implementation Phasing

1. Framework: main(), HDF5 I/O, TrialReader, DataCache
2. XCorr kernel (most complex — validates full pipeline)
3. Polarity + PSR kernels
4. AbsShift + RelShift + CAP (deferred)

## What It Does NOT Do

- Does NOT apply module weights or masks
- Does NOT aggregate misfits
- Does NOT know about convergence or strategy
- Does NOT read `config.toml`