# forward — C++ forward stage (OpenMP CPU / CUDA)

## Role

Consumes `database.h5` (preprocessed obs + per-depth/per-duration Green's functions +
`/paraspace`) and `status_{N}.h5:/trials`; recomputes each misfit kernel per
trial and writes **intermediate products** to `status_{N}.h5:/intermediates/`
(never final misfits — Julia `assess.jl` owns semantics). Stateless per run.

## Layout

```
forward/
  CMakeLists.txt          (exports compile_commands.json)
  src/
    main.cpp              entry: load, compute, write
    data_cache.{h,cpp}    HDF5 data layout, window/lag clamps
    hdf5_io.{h,cpp}       HDF5 helpers
    mt_utils.{h,cpp}      SDR -> MT (NED)
    kernels/              shared XCorr work-item + OpenMP/CUDA launchers
    backends/             CUDA runtime, reusable buffers, batch planner
    validation.*          preflight and checked size arithmetic
    intermediates_transaction.*  atomic HDF5 replacement/recovery
  build/forward           binary (CMake build dir)
```

## Conventions

- C-order arrays: `cc_max[N_phases × N_trials]`; HDF5.jl reads as
  `(N_trials, N_phases)`.
- PSR adds center-lag `syn_energy = mᵀGᵀGm`; normalized polarity adds
  `amp_scale=(max-min)/2` and the sign of the earlier min/max extremum.
- maxlag derived from the validated common XcorrP/XcorrS config, clamped to
  `(window_len-1)/2` in DataCache; reduction loop uses the clamped stride.
- station indices 0-based internally, 1-based in HDF5.
- cache key is `(freq_idx, depth_idx, duration_idx)`; duration selects precomputed Gaussian STF σ.
- `/intermediates` uses temporary + backup groups for atomic replacement and
  startup recovery. A failed preflight/kernel/write preserves the previous final group.
- CUDA reuses maximum-combo input buffers and maximum-batch MT/output buffers for
  the whole run. Default stream is deliberately synchronous; no async pipeline.

## Build / verify

```bash
cmake -S forward -B forward/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build forward/build -j8
./forward/build/forward <database.h5> <status_N.h5>

CUDACXX=/path/to/nvcc cmake -S forward -B forward/build-cuda \
    -DFM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build forward/build-cuda -j8
./forward/build-cuda/forward --backend cuda --cuda-batch-trials 10000 \
    <database.h5> <status_N.h5>
```

CLI `--backend` overrides `FM_FORWARD_BACKEND`; `--cuda-batch-trials` overrides
`FM_CUDA_BATCH_TRIALS`. `auto` selects CUDA when available, falls back only for
not-compiled/no-device, and propagates other initialization errors.

Exact-match contract vs Julia reference is covered by
`tests/stages/forward_test.jl` (independent P+S Julia recomputation, ≤1e-9).
Set `FM_TEST_CUDA=1` for CUDA parity and `FM_TEST_SANITIZER=1` for memcheck.
