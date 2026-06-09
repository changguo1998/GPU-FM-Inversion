# Architecture

## Data flow

```
SAC waveform files (*.sac)
    │
    ▼
preprocess.jl
    │  Reads SAC headers, phase picks, event config
    │  Configures misfit parameters (band, lag, trim)
    │  Builds `env` dict: {algorithm, event, stations, dataroot}
    │  Calls JuliaSourceMechanism.loaddata!(env)
    │  Reads travel times from Green's function library
    │  Writes auto.jld2
    ▼
inverse.jl
    │  Loads auto.jld2
    │  Selects misfit modules from algorithm config
    │
    ├── calcgreen!(env)         # Parallel: Threads.@threads per station
    │       │
    │       └── Green._calgreenfun_dwn_station() or Green._calgreenfun_sem_station()
    │               │
    │               ├── DWN.dwn() → freqspec2timeseries() → waveform
    │               └── SeismicRayTrace.raytrace_fastest() → tp, ts
    │
    ├── inverse_focalmech!(env, misfits)
    │       │
    │       ├── loadtp!(env)          # Load travel times into phases
    │       ├── preprocess!(env, misfits)  # Per-module preprocess!
    │       └── inverse!(env, misfits, Grid)  # Core grid search
    │               │
    │               ├── Grid.newparameters() → list of (strike,dip,rake)
    │               ├── dc2ts() → moment tensor (6 components)
    │               └── Threads.@threads over (parameter × phase-station) pairs
    │                       │
    │                       └── misfit(phase, moment_tensor)
    │                               │
    │                               ├── XCorr.misfit()    → cross-correlation
    │                               ├── Polarity.misfit() → polarity match
    │                               ├── PSR.misfit()      → P/S amplitude ratio
    │                               ├── DTW.misfit()      → dynamic time warping
    │                               ├── AbsShift.misfit() → absolute shift
    │                               └── RelShift.misfit() → relative shift
    │
    ├── Frequency test (bootstrap)
    │       Random perturbation of best mechanism
    │       Re-run inverse_focalmech!()
    │       Converge when strike std < sin(5°) AND dip/rake std < 2.5°
    │
    ├── Stage 2: Depth refinement
    │       inverse_depth() over [step2_min_depth, step2_max_depth]
    │       Iterate until depth change < threshold
    │
    ├── Channel reselection
    │       reselect_channel() by cross-correlation threshold
    │       Adaptive threshold based on station count
    │
    └── Stage 3: Final depth refinement
            inverse_depth() over [step3_min_depth, step3_max_depth]
            Write result_stage3.jld2
```

## Module hierarchy

```
fminv/ (scripts, NOT a package)
  │
  ├── JuliaSourceMechanism.jl (package)
  │     ├── Green.jl (Green module)
  │     │     ├── DWN method: DWN.jl package
  │     │     └── SEM/FD method: glib binary format + Mmap
  │     ├── mathematics.jl
  │     │     ├── dc2ts(): SDR → moment tensor
  │     │     ├── inverse!(): main grid search loop
  │     │     └── CAPmethod!(): CAP-specific search
  │     ├── system.jl: config, SAC parsing, station setup
  │     ├── VelocityModel.jl: Crust1.0 reader
  │     ├── misfits/: XCorr, Polarity, PSR, DTW, AbsShift, RelShift, CAP
  │     └── searchingMethod/Grid.jl: parameter grid management
  │
  ├── DWN.jl (package)
  │     └── Discrete Wavenumber Method waveform synthesis
  │
  ├── SeisTools.jl (package)
  │     ├── DataProcess.jl: resample, filter, cut, detrend, taper
  │     ├── SAC.jl: SAC binary format I/O
  │     ├── Geodesy.jl: distance, azimuth calculations
  │     └── QualityControl.jl: amplitude checks
  │
  └── SeismicRayTrace.jl (package)
        └── raytrace(): travel time in layered media
```

## Key data structures

### env (Setting = Dict{String,Any})
```julia
env["algorithm"]  → {misfit, weight, searchdepth, minimum_stations, ...}
env["event"]      → {longitude, latitude, depth, magnitude, origintime, phase}
env["stations"]   → [{network, station, component,
                       meta_lon, meta_lat, meta_el, meta_dt, meta_btime,
                       base_distance, base_azimuth, base_record, base_trim,
                       green_fun, green_dt, green_model, green_tsource,
                       phases: [{type, at, tt, xcorr_band, xcorr_dt, ...}]
                      }, ...]
env["dataroot"]   → path string
```

### Green's function format
- Per-station, per-component (.gf files)
- 6-column matrix: [Mxx, Myy, Mzz, Mxy, Mxz, Myz] time series
- Metadata in file header: modelname, network, station, depth, distance, azimuth, dt, tp, ts

### Moment tensor (6 components, NED coordinate)
```
M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
```
Derived from strike(φ), dip(δ), rake(λ) via `dc2ts()` (mathematics.jl:67-78)

### Misfit module interface
Every misfit module must export:
- `tags`: Tuple of String identifiers matching config
- `properties`: Vector of required phase-level config keys
- `weight(p, s, e)`: Return weight for this phase/station/env
- `preprocess!(p, s, e)`: Set up phase data (filter, trim, resample)
- `misfit(p, m)`: Compute misfit between observed and synthetic (m = moment tensor)

## Entry points

| File | Function | Role |
|------|----------|------|
| preprocess.jl | (script body) | Data loading and prep |
| inverse.jl | `inverse_focalmech!()` | Main multi-stage pipeline |
| multistage_lib.jl | `inverse_focalmech!()` | Coarse-to-fine grid search |
| multistage_lib.jl | `inverse_depth()` | Depth parameter search |
| mathematics.jl | `inverse!()` | Core grid search loop |
| mathematics.jl | `dc2ts()` | SDR → moment tensor conversion |
| Green.jl | `calcgreen!()` | Green function computation |