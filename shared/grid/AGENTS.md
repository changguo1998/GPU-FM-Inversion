# AGENTS.md — Grid module (`shared/grid/src/`)

## Role

Trial generation from strategy parameters + grid refinement based on best-trial results. Two source files: `trial_gen.jl` and `grid_refinement.jl`.

Used by: `preprocess.jl`, `assess.jl`.

## Types

| Struct | File | Fields | Notes |
|----------------|----------------------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `GridStrategy` | `trial_gen.jl` | `strike0, dstrike, nstrike, dip0, ddip, ndip, rake0, drake, nrake, depth_indices, freq_indices, best_depth_index` | Subset of IO.Strategy — only grid-relevant fields |
| `TrialSet` | `trial_gen.jl` | `strike, dip, rake, depth, depth_idx, freq_idx` | Column vectors, length N_trials |
| `TrialResult` | `grid_refinement.jl` | `sdr[3], depth_idx, freq_idx, misfit, depth_misfits[], freq_misfits[]` | Best-trial result for refinement |

Note: `Grid.TrialSet` and `IO.TrialSet` are separate structs with identical fields. `preprocess.jl` converts between them.

## Exports

| Function | Used by | Role |
|--------------------------------------------------------------|-----------------|---------------------------------------------------------------|
| `generate_trials(strategy::GridStrategy, depth_vals)` | `preprocess.jl` | Cartesian product of axes: strike × dip × rake × depth × freq |
| `refine_strategy(current::H5IO.Strategy, best::TrialResult)` | `assess.jl` | Compute next iteration's grid from best trial |
| `prompt_operator(best_sdr, misfit, current)` | `assess.jl` | Show best result, ask continue? Returns Bool |
| `TrialResult` | `assess.jl` | Struct for best-trial data |

## Grid refinement rules

- Center: best trial SDR
- Step sizes: halved (`old_step / 2`)
- Grid size: fixed 3×3×3 SDR (`nstrike=3, ndip=3, nrake=3`)
- Depth subset: indices where `depth_misfit ≤ 1.2 × best_depth_misfit`
- Frequency subset: indices where `freq_misfit ≤ 1.2 × best_freq_misfit`
- Empty subset fallback: single best index
- Depth misfit accumulator: element-wise min across iterations
- Returns new `H5IO.Strategy` with `converged=0`, iteration incremented

## Operator prompt

- Displays best SDR + misfit + current grid description
- Reads stdin `y/N` — `true` on "y"/"Y", `false` otherwise
- Testable via `io_in`/`io_out` keyword arguments (defaults: stdin/stdout)

## Axis expansion

`expand_axis(var0, dvar, n)` → `n` values `var0 + i*dvar` for `i = 0:n-1`. If `n ≤ 0`, returns `[var0]` (fixed axis).
