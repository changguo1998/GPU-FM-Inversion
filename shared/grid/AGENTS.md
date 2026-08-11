# AGENTS.md — Grid module (`shared/grid/src/`)

## Role

Trial generation from strategy parameters + grid refinement based on best-trial results. Two source files: `trial_gen.jl` and `grid_refinement.jl`.

Used by: `preprocess.jl`, `assess.jl`.

## Types

| Struct | File | Fields | Notes |
|---------------|----------------------|------------------------------------------------------------------------|----------------------------------|
| `TrialResult` | `grid_refinement.jl` | `sdr[3], depth_idx, freq_idx, misfit, depth_misfits[], freq_misfits[]` | Best-trial result for refinement |

No separate grid/trial structs — `generate_trials` consumes the full
`IO.Strategy` (12 fields: SDR grid + depth/freq indices + iteration) and
returns `IO.TrialSet`.

## Exports

| Function | Used by | Role |
|--------------------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------|
| `generate_trials(strategy::IO.Strategy)` | `preprocess.jl` | Cartesian product of per-axis 1-based indices: strike_idx × dip_idx × rake_idx × depth_idx × freq_idx (no physical values) |
| `refine_strategy(current::H5IO.Strategy, best::TrialResult)` | `assess.jl` | Compute next iteration's grid from best trial |
| `prompt_operator(best_sdr, misfit, current)` | `assess.jl` | Show best result, ask continue? Returns Bool |
| `TrialResult` | `assess.jl` | Struct for best-trial data |

## Grid refinement rules

- Center: best trial SDR (3-wide axis `[best-step, best, best+step]`; strike wrapped mod 360, dip/rake clamped to domain)
- Step sizes: halved (`old_step / 2`)
- Grid size: fixed 3×3×3 SDR (`nstrike=3, ndip=3, nrake=3`)
- Depth subset: indices where `depth_misfit ≤ 1.2 × best_depth_misfit`
- Frequency subset: indices where `freq_misfit ≤ 1.2 × best_freq_misfit`
- Empty subset fallback: single best index
- Depth misfit accumulator: element-wise min across iterations
- Returns new `H5IO.Strategy` with iteration incremented

## Operator prompt

- Displays best SDR + misfit + current grid description
- Reads stdin `y/N` — `true` on "y"/"Y", `false` otherwise
- Testable via `io_in`/`io_out` keyword arguments (defaults: stdin/stdout)

## Axis expansion

`expand_axis(var0, dvar, n)` → `n` values `var0 + i*dvar` for `i = 0:n-1`. If `n ≤ 0`, returns `[var0]` (fixed axis).
