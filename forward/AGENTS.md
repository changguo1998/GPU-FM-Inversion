# AGENTS.md — Forward module (`forward/`)

## Role

C++ executable for GPU misfit computation. Reads preprocessed data from `database.h5` and trial parameters from `status_{N}.h5`, computes raw per-module misfits, writes results. Stateless — no weights, no aggregation, no strategy knowledge. Runs once per pipeline iteration.

Entry point: `forward/src/main.cpp` → compiled to `forward/build/forward`.

## Build

```
cmake -B build -S forward
cmake --build build
```

Requires: C++17, OpenMP (required), HDF5 (serial, Ubuntu paths: `/usr/include/hdf5/serial`), CUDA (optional — compile with `nvcc` for GPU support).

## Targets (CMakeLists.txt)

| Target                | Sources                                                  | Links        |
|-----------------------|----------------------------------------------------------|--------------|
| `forward`             | `main.cpp, mt_utils.cpp, hdf5_io.cpp, data_cache.cpp`    | OpenMP, HDF5 |
| `test_hdf5_roundtrip` | `tests/test_hdf5_roundtrip.cpp, hdf5_io.cpp`             | HDF5         |
| `test_mt_utils`       | `tests/test_mt_to_csv.cpp, mt_utils.cpp`                 | —            |
| `test_data_cache`     | `tests/test_data_cache.cpp, hdf5_io.cpp, data_cache.cpp` | OpenMP, HDF5 |
| `test_xcorr`          | `tests/test_xcorr.cpp`                                   | OpenMP       |
| `test_misfit_kernels` | `tests/test_misfit_kernels.cpp`                          | OpenMP       |
| `test_cross_lang`     | `tests/test_cross_lang.cpp, mt_utils.cpp, hdf5_io.cpp`   | HDF5         |

## Source map

| File                            | Role                                                                                                                                                         |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/main.cpp`                  | Entry point. Read trials, SDR→MT, iterate (freq,depth) combos, launch kernels, write misfits                                                                 |
| `src/backends/device.h`         | Backend dispatch: `Device<OpenMP>` = `#pragma omp parallel for`; `Device<CUDA>` = `__global__` kernel. `DefaultDevice` = CUDA when `__CUDACC__`, else OpenMP |
| `src/kernels/xcorr_kernel.h`    | XCorr misfit kernel. Formula: `1.0 - max_k(|cc_syn[k] / √(obs_norm² · syn_norm²)|)`                                                                          |
| `src/kernels/polarity_kernel.h` | Polarity misfit kernel. Formula: `sign(dot(pol_vec, mt)) == obs_pol ? 0.0 : 1.0`                                                                             |
| `src/kernels/psr_kernel.h`      | PSR misfit kernel. Formula: `(log10(√(mᵀ·amp_P·m) / √(mᵀ·amp_S·m)) - obs_psr)²`                                                                              |
| `src/hdf5_io.h/.cpp`            | `Hdf5Handle` struct: open/close, read scalar/1D/2D, write 2D, group ops                                                                                      |
| `src/data_cache.h/.cpp`         | `DataCache` class: loads preprocessed obs/GF from database.h5, computes CC(obs, GF[:,i]) reductions, synamp, obs_norm2 per (freq,depth) combo                |
| `src/mt_utils.h/.cpp`           | `sdr_to_mt(radians)` → `MomentTensor{Mxx,Myy,Mzz,Mxy,Mxz,Myz}`. Identical formulas to Julia `MT.jl`.                                                         |

## Kernel details

### XCorr (`launch_xcorr_misfit`)
Work item per (phase × trial). Per item:
1. Load 6-component MT for this trial
2. Compute `syn_norm² = mᵀ · synamp[phase] · m` (6×6 quadratic form)
3. For each lag `k`: `cc_syn[k] = Σᵢ m[i] · CC[phase][k][i]`
4. Normalize: `cc_norm[k] = cc_syn[k] / √(obs_norm²[phase] · syn_norm²)`
5. Misfit = `1.0 - max_k(|cc_norm[k]|)`
Guarded: zero/negative norms → misfit = 1.0.

### Polarity (`launch_polarity_kernel`)
Work item per (station × trial). Per item:
1. Compute `dot = Σᵢ pol_vec[station][i] · mt[trial][i]`
2. `syn_pol = sign(dot)` (1/-1/0)
3. Match check: `syn_pol == obs_pol ? 0.0 : 1.0`
Guarded: NaN obs_pol or zero obs_pol with zero pol_vec → NaN (not applicable).

### PSR (`launch_psr_kernel`)
Work item per (station × trial). Per item:
1. Compute `amp_P_quad = Σᵢⱼ amp_P[station][i][j] · mt[trial][i] · mt[trial][j]`
2. Same for amp_S
3. `syn_amp = √(quad)`, `syn_psr = log10(syn_amp_P / syn_amp_S)`
4. Misfit = `(syn_psr - obs_psr)²`
Guarded: NaN obs_psr or near-zero amplitudes → NaN. Negative quads clamped to 0.

## Data flow in main.cpp

1. Read trials from `status_{N}.h5` → `vector<Trial>`
2. SDR→MT conversion (host): produces two layouts — `mt_xcorr_host[6 × N_trials]` and `mt_pol_host[N_trials × 6]`
3. Read phase index from `database.h5` → phase types, station→phase mapping
4. `DataCache::load_from_database(database_path, trials)` → loads preprocessed data per combo
5. For each (freq_idx, depth_idx) combo:
   - Build MT sub-views for combo's trials
   - Launch XCorr kernel (if XCorr data exists)
   - Launch Polarity kernel (if Polarity data exists)
   - Launch PSR kernel (if PSR data exists)
   - Write results back to global output arrays
6. Write `/misfits/xcorr`, `/misfits/polarity`, `/misfits/psr` to `status_{N}.h5`
7. Free GPU memory via `cache.release_all()`

## Memory layout conventions

All flat `double*` arrays, column-major:
- MT (XCorr): `[6 × N_trials]` — `mt[comp + trial * 6]`
- MT (Polarity/PSR): `[N_trials × 6]` — `mt[trial + comp * N_trials]`
- CC data: `[N_phases·cc_pp × 6]` — `cc[phase*cc_pp + lag + comp * (N_ph·cc_pp)]`
- Synamp: `[N_phases × 36]` — `synamp[phase + (i*6+j) * N_phases]`
- Misfit output: `[N_phases × N_trials]` — `misfit[phase + trial * N_phases]`
- Polarity/PSR misfit: `[N_stations × N_trials]` — `misfit[station + trial * N_stations]`

## Tests

| Test                  | What it verifies                                                |
|-----------------------|-----------------------------------------------------------------|
| `test_hdf5_roundtrip` | Write then read scalar/1D/2D datasets, group ops                |
| `test_mt_to_csv`      | MT conversion matches known values                              |
| `test_data_cache`     | Cache loads combos, returns correct entry, handles missing data |
| `test_xcorr`          | XCorr kernel: known-misfit cases, edge cases (zero norms)       |
| `test_misfit_kernels` | Polarity + PSR kernels: match/mismatch cases, NaN guards        |
| `test_cross_lang`     | C++ MT output matches Julia MT.jl output                        |