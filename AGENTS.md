# AGENTS.md — Focal Mechanism Inversion (CUDA Rewrite)

## Project identity

A CUDA-accelerated focal mechanism inversion pipeline. Julia for preprocessing and strategy, C++/Kokkos for GPU misfit computation, HDF5 for data exchange between stages.

The original Julia implementation lives in `old_codes/` for reference but is not part of the new build. No build system, no CI, no tests yet — greenfield from here.

## Documentation map

| Doc | Purpose |
|-----|---------|
| `doc/design.md` | Pipeline architecture: 4-stage loop, per-iteration files, design principles |
| `doc/schema.md` | HDF5 schemas for all 4 files: datasets, types, shapes |
| `doc/stages/` | Virtual stage docs: driver, setup, forward, assess, export — role, I/O, responsibilities |
| `doc/modules/` | Implementation module docs: kernel specs, interfaces, design decisions |
| `doc/plan.md` | Phased implementation and testing plan |
| `doc/old/architecture.md` | Old Julia code data flow and module hierarchy |
| `doc/old/algorithm.md` | Old misfit functions, grid search, multi-stage workflow |
| `doc/old/parallelism.md` | Old code parallelism analysis for CUDA rewrite |
| `doc/old/dependencies.md` | Old package roles, binary formats, data paths |

## Pipeline (4 stages)

```
driver.sh loops: setup → forward → assess → [repeat on status_{N}.h5] → export
```

| Stage | Language | Role |
|-------|----------|------|
| `setup.jl` | Julia | Import `raw.h5`, preprocess all data → `database.h5`; generate trials → `status_{N}.h5` |
| `forward.cpp` | C++/Kokkos | GPU misfit: per-module, per-phase, per-trial. No weights, no aggregation |
| `assess.jl` | Julia | Apply weights, aggregate, refine grid, prompt operator → `status_{N+1}.h5` |
| `export.jl` | Julia | Compile final solution → `output.h5` |

## HDF5 files (4)

| File | Lifetime | Holds |
|------|----------|-------|
| `raw.h5` | Static | External data: event info, station metadata, raw SAC waveforms |
| `database.h5` | Static | All preprocessed data: Greens at all depths, all frequency-band variants, per-module preprocessing |
| `status_{N}.h5` | Per-iteration | Self-contained: strategy, trials, misfits for iteration N |
| `output.h5` | Final | Best-fit parameters with uncertainties |

## Domain concepts

- **Moment tensor**: 6 components in NED: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
- **Source params**: strike [0,360), dip [0,90], rake [-90,90]
- **Green's functions**: 6-component waveforms per station, pre-computed externally
- **Misfit modules**: XCorr, Polarity, PSR (primary). AbsShift, RelShift, CAP — deferred.
- **Trial**: one combination of variable parameters (SDR, depth, frequency, etc.) — indexed into precomputed data slices in `database.h5`
- **Phase** = station + channel + wave type (P/S) — channels are subsumed by phases

## Key design rules

1. `forward.cpp` is stateless — reads preprocessed data + trial params, writes raw misfits
2. `assess.jl` owns all strategy: weights, channel selection, grid refinement, and prompts operator for continue/break
3. All frequency-band variants precomputed upfront in `database.h5`
4. Misfits are unweighted, per-module shapes: XCorr `[N_ph × N_tr]`, Polarity `[N_st × N_tr]`, PSR `[N_st × N_tr]`. Weights applied in assess.
5. Green's functions pre-computed externally, loaded by `setup.jl`

## Old code (reference only)

The original Julia pipeline is in `old_codes/`. It's a set of scripts (`fminv/`) plus three packages (`JuliaSourceMechanism.jl/`, `DWN.jl/`, `SeisTools.jl/`). Run via `julia old_codes/fminv/preprocess.jl <event>` then `inverse.jl <event>`.

Key data structures from old code that inform the new schema:
- Event: `longitude, latitude, depth, magnitude, origintime`
- Station: `network, station, component, meta_lon, meta_lat, meta_el, meta_dt, base_distance, base_azimuth`
- Phase: `type (P/S), at (DateTime), tt (travel time), xcorr_band, xcorr_trim, xcorr_dt`
- Green's: 6-column matrix `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`, metadata includes `dt, tp, ts, model, depth, distance, azimuth`
- Grid search: 3-step coarse-to-fine `(11/13/14°) → (5/3/3°) → (1/1/1°)`

Full details in `doc/old/`.
