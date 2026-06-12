# Module: Data Cache (GPU)

## Description

GPU memory management and ephemeral reductions for `forward.cpp`. Loads preprocessed data from `database.h5`, computes only the reduced arrays needed for the current run, and caches reduced data on GPU.

`input.jl` owns persistent preprocessing in `database.h5` (filtering, trimming, and any stored reduced datasets). `DataCache` owns non-persistent GPU reductions and may reuse reduced datasets already stored in `database.h5`.

## Used By

- `forward.cpp` — central data management component

## Architecture

```
DataCache
├── load_from_database(h5_file, trial_set)  → populates cache
├── get_or_compute(freq_idx, depth_idx, phase_id) → returns reduced data
├── release_old_combos()  → evict unused (freq, depth) combos
└── release_all()  → free GPU memory
```

## Cache Key

`(freq_idx, depth_idx)` — all phases for this combination loaded at once.

## Reduction by Module

| Module | Input | Output per phase/station | Shape |
|--------|-------|--------------------------|-------|
| XCorr | `obs [N_samples]`, `gf [N_samples × 6]` | `cc [2·maxlag+1 × 6]`, `synamp [6×6]`, `obs_norm2 [scalar]` | ~49 KB |
| Polarity | `gf_pol [N_pol_samples × 6]` | `pol_vec [6]` | 48 B |
| PSR | `gf_P [N_P_samples × 6]`, `gf_S [N_S_samples × 6]` | `amp_P [6×6]`, `amp_S [6×6]`, `obs_psr [scalar]` | ~0.6 KB |
| CAP (deferred) | — | — | — |

**Memory budget** (typical event: 40 phases, 2 combos): ~4 MB total GPU memory.

## Execution Flow

```
1. Identify all (freq_idx, depth_idx) combos referenced by trials
2. For each combo:
   a. Read preprocessed waveforms or reduced datasets from database.h5
   b. Transfer to GPU
   c. Launch Kokkos reductions for any arrays not already persisted
   d. Discard temporary waveform arrays from GPU
   e. Keep reduced data in cache
3. All combos cached simultaneously (fit in GPU memory)
```

## Kokkos Reductions

```cpp
// Reductions are expressed as Kokkos::parallel_for / parallel_reduce work.
// They produce the module-specific reduced arrays listed above, without
// backend-specific launch syntax in the module contract.
```

## Key Design Decisions

- **Linear decomposition**: Precompute `CC(obs, GF[:,i])` once per phase. Per-trial: weighted sum of precomputed CCs. Reduces per-trial work by ~4000×.
- **Time-domain over FFT**: Simple, portable, avoids cuFFT dependency. FFT-based optimization deferred.
- **Single-pass data load**: All data loaded once to GPU → all precomputation → all kernels back-to-back. No data movement between modules.
- **No per-trial GF loading**: GF is preloaded per (freq, depth) combo, not per trial.
- **GPU memory eviction**: For large events exceeding GPU memory, evict least-recently-used combos. v1 assumes all combos fit.
