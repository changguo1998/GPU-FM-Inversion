# AGENTS.md — Config module (`shared/config/src/Config.jl`)

## Role

Pipeline configuration interface. Declares functions that the user's `config.jl` script must implement. Each unimplemented function throws a descriptive `ConfigError` at runtime.

Also provides `use_misfit!()` for loading misfit module plugins from `shared/misfit/`.

Used by: `input.jl` (via `include(config_jl)` which defines the functions).

## Exports

### Configuration functions

| Function | Return type | Example return value |
|---------------------|-----------------------------------|------------------------------------------|
| `misfit_modules()` | `Vector{String}` | Auto-detected from `use_misfit!()` calls |
| `freq_bands()` | `Vector{Tuple{Float64, Float64}}` | `[(0.5, 2.0)]` |
| `depths()` | `Vector{Float64}` | `[5.0, 10.0, 15.0]` |
| `phase_fields()` | `Dict{String, Symbol}` | `Dict("P" => :P_time, "S" => :S_time)` |
| `polarity_fields()` | `Dict{String, Symbol}` | `Dict("P" => :P_polarity)` |

### Misfit plugin loader

| Function | Description |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_misfit!(name; operator, output, phase=nothing, bases=nothing, channel=nothing)` | Register misfit instance. Level 1 (base): `operator` (Module) + `phase` ("P"/"S") + `output` (∈ `operator.outputs()`); creates `Config.{name}` instance module for per-instance overrides. Level 2 (composed): `operator` (aggregate) + `bases` (Vector{Symbol}) + `output`; no instance module. `channel` optionally filters Level 1 by channel. |
| `operator_module(name)` | Return the operator Module for an instance |
| `output_field(name)` | Return the selected output field Symbol |
| `bases_of(name)` | Return base instance names for a composed misfit |
| `is_composed(name)` | `true` if Level 2 (composed) |
| `channel_of(name)` | Return channel filter or `nothing` |
| `phase_type(name)` | Return the declared phase type, or `nothing` |

### Inner modules (loaded via `use_misfit!`)

Each loaded plugin creates a `Config.{name}` inner module. Functions depend on the plugin:

| Plugin | Functions |
|--------------------------------|------------------------------------------------|
`Misfit.Polarity` (template) — deferred (XCorr-only mode)
| `XcorrP`, `XcorrS` (instances) | Inherited from `Misfit.Xcorr` template |
`PolarityP` (instance) — deferred (XCorr-only mode)

Users instantiate operators with `Config.use_misfit!(...; operator=Misfit.Xcorr, output=Misfit.Xcorr.CC_MAX)` then override functions:

```julia
using Misfit

Config.use_misfit!(:XcorrP,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.CC_MAX)
Config.XcorrP.trim() = [-2.0, 5.0]
Config.XcorrP.max_lag_periods() = 0.5
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
using Misfit

Config.use_misfit!(:XcorrP,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.CC_MAX)
Config.use_misfit!(:XcorrS,
    operator = Misfit.Xcorr, phase = "S", output = Misfit.Xcorr.CC_MAX)
Config.use_misfit!(:PolarityP,
    operator = Misfit.Polarity, phase = "P", output = Misfit.Polarity.SYN_SIGN)
Config.XcorrP.trim() = [-2.0, 5.0]
Config.XcorrP.max_lag_periods() = 0.5
Config.XcorrP.filter_order() = 4

Config.XcorrS.trim() = [-2.0, 5.0]
Config.XcorrS.max_lag_periods() = 0.5
Config.XcorrS.filter_order() = 4

Config.PolarityP.trim() = [0.0, 2.0]

Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 10.0, 15.0]
Config.phase_fields() = Dict("P" => :P_time, "S" => :S_time)
Config.polarity_fields() = Dict("P" => :P_polarity)

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
