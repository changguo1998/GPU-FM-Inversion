# AGENTS.md — Grid module (`shared/grid/src/`)

## Role

Generate trial parameter combinations from `IO.Strategy`. Used by
`scripts/preprocess.jl`.

The previous assess refinement strategy is archived under
`archive/assess-refinement-v1/` and is not part of the active module.

## Exports

| Function | Used by | Role |
| `generate_trials(strategy::IO.Strategy)` | `preprocess.jl` | Cartesian product of strike × dip × rake × depth × frequency × duration |
| `default_grid()` | `input.jl` | Return the canonical full-space SDR grid |

## Axis expansion

`expand_axis(var0, dvar, n)` returns `n` values `var0 + i*dvar` for
`i = 0:n-1`. If `n ≤ 0`, it returns `[var0]`.
