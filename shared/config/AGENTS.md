# AGENTS.md — Config module (`shared/config/src/Config.jl`)

## Role

Pipeline configuration interface. Declares functions that the user's `config.jl` script must implement. Each unimplemented function throws a descriptive `ConfigError` at runtime.

`@objective` 是用户面目标函数接口。`input.jl` 先将表达式编译为管道算子实例，
再持久化到 `database.h5:/config/objectives`。`use_misfit!()` 保留为编译后端/兼容接口。

Used by: `input.jl` (via `include(config_jl)` which defines the functions).

## Exports

### Configuration functions

| Function | Return type | Example return value |
|---------------------|-----------------------------------|-----------------------------------------------------------------|
| `misfit_modules()` | `Vector{String}` | Auto-detected from compiled objectives or `use_misfit!()` calls |
| `freq_bands()` | `Vector{Tuple{Float64, Float64}}` | `[(0.5, 2.0)]` |
| `depths()` | `Vector{Float64}` | `[5.0, 10.0, 15.0]` |
| `durations()` | `Vector{Float64}` | Gaussian STF σ candidates in seconds, e.g. `[0.1, 0.2, 0.3]` |
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

### 目标函数表达式

```julia
p_obs = observed(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
p_syn = synthetic(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
Config.@objective XcorrP = 1 - maxCC(p_obs, p_syn; maxlag = 3)
```

`window` 和 `maxlag` 单位为主频周期数，`band` 单位为 Hz。`objective(name)`读取单个
表达式，`objectives()`返回注册表副本。`compile_objectives!()` 按注册顺序编译。
首版只支持 `1 - maxCC(observed(...), synthetic(...))` 和单一频带；P/S 共用现有 XCorr 后端。

### Inner modules (loaded via `use_misfit!`)

Each loaded plugin creates a `Config.{name}` inner module. Functions depend on the plugin:

| Plugin | Functions |
|--------------------------------|------------------------------------------------|
`Misfit.Polarity` (template) — deferred (XCorr-only mode)
| `XcorrP`, `XcorrS` (instances) | Inherited from `Misfit.Xcorr` template |
`PolarityP` (instance) — deferred (XCorr-only mode)

Compiler/backend code instantiates operators with `Config.use_misfit!()`; new user configs use `@objective`:

```julia
using Misfit

p_obs = observed(P; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
p_syn = synthetic(P; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
Config.@objective XcorrP = 1 - maxCC(p_obs, p_syn; maxlag = 0.5)
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

p_obs = observed(P; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
p_syn = synthetic(P; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
s_obs = observed(S; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
s_syn = synthetic(S; band = (0.5, 2.0), window = (-2, 5), filter_order = 4)
Config.@objective XcorrP = 1 - maxCC(p_obs, p_syn; maxlag = 0.5)
Config.@objective XcorrS = 1 - maxCC(s_obs, s_syn; maxlag = 0.5)

Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 10.0, 15.0]
Config.durations() = [0.1, 0.2, 0.3]
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

- 配置接口与轻量注册模块；不执行数值计算，不负责 HDF5 I/O。
- Each function uses `throw(ConfigError(...))` as default body (plugin templates use `Config.ConfigError`).
- Function signatures enforce return types with `::` annotations where practical.
- `input.jl` uses `include()` to evaluate config in the same scope — users register `Config.xxx()` functions directly.
- Misfit module plugins live in `shared/misfit/`; DSL compiler lowers supported expressions via `Config.use_misfit!()`.
