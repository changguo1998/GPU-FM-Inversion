# Module: Data Cache

## Description

Host-side data cache for `forward.cpp`. Loads preprocessed data from `database.h5`, computes only the reduced arrays needed for the current run, and caches results. All data is host-resident (`double*` allocated with `new[]`) — no device memory management in v1.

`input.jl` owns persistent preprocessing in `database.h5` (filtering, trimming, and any stored reduced datasets). `DataCache` owns non-persistent reductions computed from `database.h5` data.

## Used By

- `forward.cpp` — central data management component

## Architecture

```
DataCache (non-template, plain class)
├── load_from_database(h5_file, trial_set)  → populates cache
├── get_or_compute(freq_idx, depth_idx)      → returns const CacheEntry*
└── release_all()                             → frees all host memory
```

The cache is a `std::unordered_map<std::pair<int,int>, CacheEntry>` keyed by `(freq_idx, depth_idx)`.

## Cache Key

`(freq_idx, depth_idx)` — all phases/stations for this combination loaded at once.

## CacheEntry Structure

```cpp
struct CacheEntry {
    int freq_idx, depth_idx, maxlag, n_phases, n_stations;
    XCorrCache   xcorr;    // cc[N_ph·cc_pp × 6], synamp[N_ph × 36], obs_norm2[N_ph]
    PolarityCache polarity; // pol_vec[N_ph × 6], obs_pol[N_ph]
    PSRCache     psr;       // amp_P[N_ph × 36], amp_S[N_ph × 36], obs_psr[N_ph]
};
```

All fields are flat `double*` arrays allocated with `new[]` and freed by `CacheEntry::release()`.

## Reduction by Module

| Module | Input from HDF5 | Output (host-resident) | Size/phase |
|--------|-----------------|------------------------|------------|
| XCorr | `obs [N_samples]`, `gf [N_samples × 6]` | `cc [cc_pp × 6]`, `synamp [36]`, `obs_norm2` | ~0.5 KB |
| Polarity | `gf_pol [N_pol_samples × 6]`, `obs_pol` | `pol_vec [6]`, `obs_pol` | ~56 B |
| PSR | `amp_P [36]`, `amp_S [36]`, `obs_psr` | `amp_P [36]`, `amp_S [36]`, `obs_psr` | ~584 B |

**Note:** PSR `amp_P` and `amp_S` are read directly from `database.h5` (already precomputed by input.jl) — no further reduction is needed; they're copied through unchanged.

**Memory budget** (typical event: 40 phases, 2 combos): ~4 MB total.

## Execution Flow

```
1. Extract unique (freq_idx, depth_idx) combos from trial set
2. For each combo:
   a. Open database.h5
   b. Read phase_ids, phase types, station indices
   c. Read per-phase HDF5 data into temporary vectors
   d. Compute reductions on host (compute_xcorr_reduction, etc.)
   e. Discard temporary waveform vectors
   f. Store CacheEntry in unordered_map
3. Kernel launch uses cache_.at(key) to get data pointers
```

## Reductions

Reductions run on the CPU via plain loops — no GPU dispatch at this layer:

```cpp
class DataCache {
    static void compute_xcorr_reduction(CacheEntry& entry,
        const std::vector<double>& obs, const std::vector<double>& gf, int n_samples);
    static void compute_polarity_reduction(CacheEntry& entry,
        const std::vector<double>& gf_pol, int n_pol_samples);
    static void compute_psr_reduction(CacheEntry& entry,
        const std::vector<double>& ampP, const std::vector<double>& ampS,
        const std::vector<double>& obs_psr);
};
```

## Key Design Decisions

- **Host-side only**: All cache data lives on the CPU. Kernels receive raw `const double*` pointers into the cache. GPU offloading happens at the kernel level via `Device<Backend::CUDA>::parallel_for`.
- **Linear decomposition**: Precompute `CC(obs, GF[:,i])` once per phase. Per-trial: weighted sum of precomputed CCs. Reduces per-trial work by ~4000×.
- **Time-domain over FFT**: Simple, portable, avoids cuFFT dependency. FFT-based optimization deferred.
- **Single-pass data load**: All data loaded from HDF5 once → all reductions → all kernels back-to-back. No data movement between modules.
- **No per-trial GF loading**: GF is preloaded per (freq, depth) combo, not per trial.
- **v1 assumes all combos fit in host memory** — no eviction policy implemented yet.
- **No external GPU framework**: Custom `Device<Backend>` template replaces Kokkos. Only OpenMP and CUDA backends needed.
