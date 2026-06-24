# AGENTS.md — Config module (`shared/config/src/Config.jl`)

## Role

Pipeline configuration interface. Declares functions that the user's `config.jl` script must implement. Each unimplemented function throws a descriptive `ConfigError` at runtime.

Used by: `input.jl` (via `include(config_jl)` which defines the functions).

## Exports

| Function             | Return type                       | Example return value                             |
|----------------------|-----------------------------------|--------------------------------------------------|
| `data_file()`        | `String`                          | `"data/my_event.h5"`                             |
| `misfit_modules()`   | `Vector{String}`                  | `["XCorr", "Polarity"]`                          |
| `module_weights()`   | `Vector{Float64}`                 | `[0.5, 0.5]`                                     |
| `minimum_stations()` | `Int`                             | `2`                                              |
| `freq_bands()`       | `Vector{Tuple{Float64, Float64}}` | `[(0.5, 2.0)]`                                   |
| `depths()`           | `Vector{Float64}`                 | `[5.0, 10.0, 15.0]`                              |
| `grid_params()`      | NamedTuple{9}                     | `(strike0=45.0, dstrike=20.0, nstrike=3, ...)`   |
| `xcorr_params()`     | NamedTuple{6}                     | `(maxlag_factor=0.5, filter_order=4, ...)`       |
| `polarity_params()`  | NamedTuple{1}                     | `(trim=[0.0, 2.0],)`                             |
| `greens_params()`    | NamedTuple{2}                     | `(gf_dir="tests/synthetic/", model="synthetic")` |

## Error handling

`ConfigError(func_name, hint_message)` prints exactly which function is missing and what it should return:

```
ConfigError: misfit_modules() is not implemented.
  Your config script must define:  misfit_modules()  -> Vector{String}
```

## Config file pattern

User writes a `.jl` file that includes this module and implements the functions:

```julia
using Config
Config.data_file() = "data/my_event.h5"
Config.misfit_modules() = ["XCorr", "Polarity"]
# ... etc
```

The stage script (`input.jl`) loads it via `include(abspath(config_jl))`. All config values are written to `database.h5` by `input.jl`; subsequent stages never read the original config file.

## Coding conventions

- Interface-only module — no implementation logic, no HDF5 I/O.
- Each function uses `throw(ConfigError(...))` as default body.
- Function signatures enforce return types with `::` annotations.
- Not a proper Julia module with `export` — `input.jl` uses `include()` to evaluate config in the same `Config` namespace.