# Module: Grid (Trial Generation)

**Location**: `shared/grid/` (Julia package `Grid`)

Grid expands an `IO.Strategy` into the Cartesian product consumed by
`scripts/preprocess.jl`. The previous assess refinement strategy has been
archived under `archive/assess-refinement-v1/` and is no longer active.

## Input

- SDR grid start, step, and count
- `depth_indices`
- `freq_indices`
- `duration_indices`
- iteration number

## Output

`generate_trials(strategy)` returns `IO.TrialSet` with 1-based parameter-space
indices. Trial order is strike × dip × rake × depth × frequency × duration.

An axis with a non-positive count contributes its start value once. Empty
depth, frequency, or duration index vectors default to index `1`.
