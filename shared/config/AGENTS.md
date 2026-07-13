# AGENTS.md — Config module (`shared/config/src/Config.jl`)

## Role

Pipeline configuration interface. Declares functions that the user's `config.jl` script must implement. Each unimplemented function throws a descriptive `ConfigError` at runtime.

Also provides `use_misfit!()` for loading misfit module plugins from `shared/misfit/`.

Used by: `input.jl` (via `include(config_jl)` which defines the functions).

## Exports

### Configuration functions

| Function | Return type | Example return value |
|--------------------|-----------------------------------|------------------------------------------|
| `misfit_modules()` | `Vector{String}` | Auto-detected from `use_misfit!()` calls |
| `freq_bands()` | `Vector{Tuple{Float64, Float64}}` | `[(0.5, 2.0)]` |
| `depths()` | `Vector{Float64}` | `[5.0, 10.0, 15.0]` |

### Misfit plugin loader

| Function | Description |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `use_misfit!(name; from=name)` | Load plugin from `shared/misfit/{from}.jl`, create `Config.{name}` inner module, auto-register in `misfit_modules()` |

### Inner modules (loaded via `use_misfit!`)

Each loaded plugin creates a `Config.{name}` inner module. Functions depend on the plugin:

| Plugin | Functions |
|--------------------|---------------------------------------------------------------------------------------------|
| `Xcorr` (template) | `trim()`, `maxlag_factor()`, `filter_order()`, `select_threshold()`, `deselect_threshold()` |
| `Polarity` | `trim()` |

Users instantiate templates with `Config.use_misfit!(:XcorrP, from = :Xcorr)` then override functions:

```julia
Config.use_misfit!(:XcorrP, from = :Xcorr)
Config.XcorrP.trim() = [-2.0, 5.0]
Config.XcorrP.maxlag_factor() = 0.5
```

### Data interface functions

| Function | Return type | Description |
|----------------------------------------------------------|--------------------------|------------------------------------------------------------------|
| `load_event()` | `IO.EventInfo` | Event location, magnitude, origin time |
| `load_stations()` | `Vector{IO.StationInfo}` | Station metadata for all channels |
| `load_phase_picks()` | `Vector{IO.PhasePick}` | P/S arrival times and P polarity |
| `load_waveform(phase_id)` | `Vector{Float64}` | Raw observed waveform for a phase |
| `load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon)` | `Union{Nothing, Tuple}` | GF array `(nt,6,3)`, dt, tp, ts; return `nothing` if unavailable |

## Error handling

`ConfigError(func_name, hint_message)` prints exactly which function is missing and what it should return:

```
ConfigError: Xcorr.trim() is not implemented.
  Your config script must define:  Xcorr.trim()  -> Vector{Float64}
```

## Config file pattern

User writes a `.jl` file that implements the functions:

```julia
# (Config module is already loaded by input.jl)
Config.use_misfit!(:XcorrP, from = :Xcorr)
Config.use_misfit!(:XcorrS, from = :Xcorr)
Config.use_misfit!(:Polarity)

Config.XcorrP.trim() = [-2.0, 5.0]
Config.XcorrP.maxlag_factor() = 0.5
Config.XcorrP.filter_order() = 4
Config.XcorrP.select_threshold() = 0.5
Config.XcorrP.deselect_threshold() = 0.3

Config.XcorrS.trim() = [-2.0, 5.0]
Config.XcorrS.maxlag_factor() = 0.5
Config.XcorrS.filter_order() = 4
Config.XcorrS.select_threshold() = 0.5
Config.XcorrS.deselect_threshold() = 0.3

Config.Polarity.trim() = [0.0, 2.0]

Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 10.0, 15.0]

# Data reading
Config.load_event() = IO.EventInfo(120.0, 30.0, 10.0, 5.0, "2024-01-01T00:00:00")
Config.load_stations() = begin ... end
Config.load_phase_picks() = begin ... end
Config.load_waveform(pid) = begin ... end
Config.load_gf(...) = begin ... end
```

The stage script (`input.jl`) loads it via `include(abspath(config_jl))`. All config values are written to `database.h5` by `input.jl`; subsequent stages never read the original config file.

## Coding conventions

- Interface-only module — no implementation logic, no HDF5 I/O.
- Each function uses `throw(ConfigError(...))` as default body (plugin templates use `Config.ConfigError`).
- Function signatures enforce return types with `::` annotations where practical.
- `input.jl` uses `include()` to evaluate config in the same scope — users register `Config.xxx()` functions directly.
- Misfit module plugins live in `shared/misfit/` and are loaded via `Config.use_misfit!()`.
