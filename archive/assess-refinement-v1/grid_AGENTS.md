# Archived AGENTS.md — Grid module (`shared/grid/src/`)

## Role

Trial generation from strategy parameters + grid refinement based on best-trial
results. Two source files: `trial_gen.jl` and `grid_refinement.jl`.

Used by: `preprocess.jl`, `assess.jl`.

## Types

| Struct | File | Fields | Notes |
| `TrialResult` | `grid_refinement.jl` | `sdr[3], depth_idx, freq_idx, misfit, depth_misfits[], freq_misfits[]` | Best-trial result for refinement |

## Exports

| Function | Used by | Role |
| `generate_trials(strategy::IO.Strategy)` | `preprocess.jl` | Cartesian product of parameter indices |
| `refine_strategy(current, best)` | `assess.jl` | Compute next iteration grid |
| `prompt_operator(best_sdr, misfit, current)` | `assess.jl` | Ask whether to continue |

## Grid refinement rules

- Center on best trial SDR using a three-wide axis
- Halve SDR step sizes
- Use a fixed `3×3×3` SDR grid
- Keep depth/frequency indices within `1.2 ×` their best misfit
- Preserve duration indices
- Fall back to the single best index for an empty subset
- Increment the iteration number

## Operator prompt

Read `y/N` from standard input and continue only for `y` or `Y`.
