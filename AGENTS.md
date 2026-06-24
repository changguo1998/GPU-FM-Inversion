# AGENTS.md — Focal Mechanism Inversion (CUDA Rewrite)

## Project identity

A CUDA-accelerated focal mechanism inversion pipeline. Julia for preprocessing and strategy, C++ with custom OpenMP/CUDA backend for GPU misfit computation, HDF5 for data exchange between stages.

The original Julia implementation has been removed from the working tree. Historical structure and behavior notes are preserved under `doc/old/`. The current rewrite has a CMake build for `forward/`, stage-local Julia projects, and focused unit/integration tests; CI is not set up yet.

## Documentation map

| Doc | Purpose |
|-----|---------|
| `doc/design.md` | Pipeline architecture: 5-stage pipeline, per-iteration files, design principles |
| `doc/schema.md` | HDF5 schemas for all 4 files: datasets, types, shapes |
| `doc/stages/` | Virtual stage docs: driver, input, preprocess, forward, assess, output — role, I/O, responsibilities |
| `doc/modules/` | Implementation module docs: kernel specs, interfaces, design decisions |
| `doc/plan.md` | Phased implementation and testing plan |
| `doc/old/architecture.md` | Old Julia code data flow and module hierarchy |
| `doc/old/algorithm.md` | Old misfit functions, grid search, multi-stage workflow |
| `doc/old/parallelism.md` | Old code parallelism analysis for CUDA rewrite |
| `doc/old/dependencies.md` | Old package roles, binary formats, data paths |

## Pipeline (5 stages)

```
driver.sh: input (once) → loop: [preprocess → forward → assess → [repeat]] → output
```

| Stage | Language | Role |
|-------|----------|------|
| `input.jl` | Julia | Read `config.jl`, locate external data, preprocess all data → `database.h5`; initial strategy → `status_0.h5` |
| `preprocess.jl` | Julia | Generate trials from strategy → `status_{N}.h5` |
| `forward.cpp` | C++ (OpenMP/CUDA) | GPU misfit: per-module, per-phase, per-trial. No weights, no aggregation |
| `assess.jl` | Julia | Apply weights, aggregate, refine grid, prompt operator → `status_{N+1}.h5` |
| `output.jl` | Julia | Compile final solution → `output.h5` |

## HDF5 files (4)

| File | Lifetime | Holds |
|------|----------|-------|
| `database.h5` | Static | All preprocessed data: Greens at all depths, all frequency-band variants, per-module preprocessing |
| `status_{N}.h5` | Per-iteration | Self-contained: strategy, trials, misfits for iteration N |
| `output.h5` | Final | Best-fit parameters with uncertainties |

## Domain concepts

- **Moment tensor**: 6 components in NED: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
- **Source params**: strike [0,360), dip [0,90], rake [-90,90]
- **Green's functions**: 6-component waveforms per station, pre-computed externally
- **Misfit modules**: XCorr, Polarity, PSR (primary). AbsShift, RelShift — deferred. CAP — cancelled.
- **Trial**: one combination of variable parameters (SDR, depth, frequency, etc.) — indexed into precomputed data slices in `database.h5`
- **Phase** = station + channel + wave type (P/S) — channels are subsumed by phases

## Key design rules

1. `forward.cpp` is stateless — reads preprocessed data + trial params, writes raw misfits. Uses custom backend dispatch (OpenMP for CPU, CUDA for GPU) rather than Kokkos.
2. `assess.jl` owns all strategy: weights, channel selection, grid refinement, and prompts operator for continue/break
3. All frequency-band variants precomputed upfront in `database.h5` by `input.jl`
4. Misfits are unweighted, per-module shapes: XCorr `[N_ph × N_tr]` (phase-level), Polarity `[N_ch × N_tr]` (channel-level P-polarity), PSR `[N_ch × N_tr]` (channel-level P/S ratio). Weights applied in assess.
5. Green's functions pre-computed externally, loaded by `input.jl`

## Old code (reference only)

The original Julia pipeline is no longer kept in-tree. Its former layout was a set of scripts (`fminv/`) plus packages such as `JuliaSourceMechanism.jl/`, `DWN.jl/`, and `SeisTools.jl/`; relevant historical notes are preserved in `doc/old/`.

Key data structures from old code that inform the new schema:

- Event: `longitude, latitude, depth, magnitude, origintime`
- Station: `network, station, component, meta_lon, meta_lat, meta_el, meta_dt, base_distance, base_azimuth`
- Phase: `type (P/S), at (DateTime), tt (travel time), xcorr_band, xcorr_trim, xcorr_dt`
- Green's: 6-column matrix `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`, metadata includes `dt, tp, ts, model, depth, distance, azimuth`
- Grid search: 3-step coarse-to-fine `(11/13/14°) → (5/3/3°) → (1/1/1°)`

Full details in `doc/old/`.
