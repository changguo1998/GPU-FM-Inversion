# forward — C++ forward stage (OpenMP CPU)

## Role

Consumes `database.h5` (preprocessed obs + per-depth Green's functions +
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
    kernels/              xcorr_kernel.h / psr_kernel.h / polarity_kernel.h
    backends/device.h     device traits
  build/forward           binary (CMake build dir)
```

## Conventions

- C-order arrays: `cc_max[N_phases × N_trials]`; HDF5.jl reads as
  `(N_trials, N_phases)`.
- maxlag derived from config `max_lag_periods` (via `/config/XcorrS`
  currently — scoped, see `doc/stages/forward.md`), clamped to
  `(window_len-1)/2` in DataCache; reduction loop uses the clamped stride.
- station indices 0-based internally, 1-based in HDF5.
- `/intermediates` rewritten idempotently each run (delete + recreate).

## Build / verify

```bash
cmake -S forward -B forward/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build forward/build -j8
./forward/build/forward <database.h5> <status_N.h5>
```

Exact-match contract vs Julia reference is covered by
`tests/stages/forward_test.jl` (independent Julia recomputation, ≤1e-9).
