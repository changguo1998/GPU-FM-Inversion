# AGENTS.md — Search module (`shared/search/src/`)

## Role

Define search-space axes and generate trial parameter combinations from
`IO.Strategy`. Used by `scripts/input.jl` and `scripts/preprocess.jl`.

The previous Grid-based assess refinement strategy is archived under
`archive/assess-refinement-v1/` and is not part of the active module.

## Exports

| Function | Used by | Role |
| `generate_trials(strategy::IO.Strategy)` | `preprocess.jl` | Cartesian product of strike × dip × rake × depth × frequency × duration |
| `default_grid()` | `input.jl` | Return the canonical full-space SDR grid |
| `budgeted_plan(strategy, budget)` | assess/search planning | Select a deterministic coarse trial set within a budget |
| `generate_trials(plan::SearchPlan)` | search planning | Materialize selected global indices as `IO.TrialSet` |

## Axis expansion

`expand_axis(var0, dvar, n)` returns `n` values `var0 + i*dvar` for
`i = 0:n-1`. If `n ≤ 0`, it returns `[var0]`.
