# AGENTS.md — Config module (`shared/config/src/Config.jl`)

## Role

Pipeline configuration interface. Declares functions that the user's `config.jl` script must implement. Each unimplemented function throws a descriptive `ConfigError` at runtime.

Used by: `input.jl` (via `include(config_jl)` which defines the functions).

## Exports

### Configuration functions

| Function | Return type | Example return value |
|----------------------|-----------------------------------|--------------------------------------------|
| `misfit_modules()` | `Vector{String}` | `["XCorr", "Polarity"]` |
| `module_weights()` | `Vector{Float64}` | `[0.5, 0.5]` |
| `minimum_stations()` | `Int` | `2` |
| `freq_bands()` | `Vector{Tuple{Float64, Float64}}` | `[(0.5, 2.0)]` |
| `depths()` | `Vector{Float64}` | `[5.0, 10.0, 15.0]` |
| `xcorr_params()` | NamedTuple{6} | `(maxlag_factor=0.5, filter_order=4, ...)` |
| `polarity_params()` | NamedTuple{1} | `(trim=[0.0, 2.0],)` |

### Data interface functions

| Function | Return type | Description |
|----------------------------------------------------------|--------------------------|------------------------------------------------------------------|
| `load_event()` | `IO.EventInfo` | Event location, magnitude, origin time |
| `load_stations()` | `Vector{IO.StationInfo}` | Station metadata for all channels |
| `load_phase_picks()` | `Vector{IO.PhasePick}` | P/S arrival times and P polarity |
| `load_waveform(phase_id)` | `Vector{Float64}` | Raw observed waveform for a phase |
| `load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon)` | `Union{Nothing, Tuple}` | GF array `(nt,6,3)`, dt, tp, ts; return `nothing` if unavailable |

The initial search grid is automatically provided by the `Grid` module (`Grid.default_grid()`).
Green's function loading is supported via `Config.load_gf()` (implemented in config.jl, exported by Config module).

## Error handling

`ConfigError(func_name, hint_message)` prints exactly which function is missing and what it should return:

```
ConfigError: misfit_modules() is not implemented.
  Your config script must define:  misfit_modules()  -> Vector{String}
```

## Config file pattern

User writes a `.jl` file that implements the functions:

```julia
# (Config module is already loaded by input.jl)
Config.misfit_modules() = ["XCorr", "Polarity"]
Config.module_weights() = [0.5, 0.5]
Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 10.0, 15.0]

# Data reading — implement your own data source
Config.load_event() = begin
    # Read from SAC, SEED, custom HDF5, ...
    IO.EventInfo(120.0, 30.0, 10.0, 5.0, "2024-01-01T00:00:00")
end
Config.load_stations() = begin
    # ...
end
Config.load_phase_picks() = begin
    # ...
end
Config.load_waveform(pid) = begin
    # ...
end
```

The stage script (`input.jl`) loads it via `include(abspath(config_jl))`. All config values are written to `database.h5` by `input.jl`; subsequent stages never read the original config file.

## Coding conventions

- Interface-only module — no implementation logic, no HDF5 I/O.
- Each function uses `throw(ConfigError(...))` as default body.
- Function signatures enforce return types with `::` annotations where practical.
- `input.jl` uses `include()` to evaluate config in the same scope — users register `Config.xxx()` functions directly.
