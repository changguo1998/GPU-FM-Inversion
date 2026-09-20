# Module: Search

**Location**: `shared/search/` (Julia package `Search`)

Search defines parameter axes and expands an `IO.Strategy` into the Cartesian
product consumed by `scripts/preprocess.jl`. The previous Grid-based assess
refinement strategy has been archived under `archive/assess-refinement-v1/`
and is no longer active.

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

## Budget-constrained planning

`budgeted_plan(strategy, budget)` returns a `SearchPlan` without changing the
fixed parameter space. It starts from boundary samples on ordered axes and two
opposite samples on periodic strike, then greedily bisects the axis with the
largest normalized resolution improvement. A refinement is accepted only when
the resulting Cartesian product remains within `budget`.

`generate_trials(plan)` materializes the selected global indices. The current
pipeline still calls `generate_trials(strategy)` and therefore retains its
full-search behavior until assess integration is designed.
