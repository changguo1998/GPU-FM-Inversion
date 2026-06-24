# Stage: `scripts/assess.jl` — Weighting, Aggregation & Strategy Update

## Role

Reads raw misfits from `status_{N}.h5`, applies module weights and channel/phase masks, aggregates into per-trial scores, refines the search grid using `shared/grid/`, and prompts the operator whether to continue or break.

**Owns all strategy decisions**: module weights, phase/channel selection, grid refinement. `forward.cpp` never applies weights.

## Inputs

| Source | Description |
|--------|-------------|
| `status_{N}.h5` | Reads `/trials`, `/misfits`, `/strategy` |
| `database.h5` | Reads `/config`, `/index` (for reference) |

## Outputs

| Source | Description |
|--------|-------------|
| `status_{N}.h5` (on break) | Sets `/strategy/converged=1`, `convergence_reason="user"` — no new file created |
| `status_{N+1}.h5` (on continue) | New file with `/strategy` (refined grid, masks, weights, best result, converged=0) |

## Responsibilities

1. **Apply per-module masks**: each module has its own mask (XCorr: phase-level, Polarity: channel-level)
2. **Weight and aggregate**: sum masked misfits per module, apply module weights, combine → per-trial total. PSR misfits are optional (zeros fallback if missing)
3. **Find best trial**: argmin of aggregated misfit
4. **Refine grid**: compute next search grid from current best SDR (halve step sizes, 3×3×3 fixed)
5. **Prompt operator**: display current results, ask "Continue? [y/N]"
6. **Signal driver via exit code**:
   - On continue (y): exit 0 — driver loops to next iteration
   - On break (N): exit 10 — driver breaks loop and runs output.jl
7. **Write strategy**:
   - On continue: create `status_{N+1}.h5` with `/strategy` (refined grid, converged=0)
   - On break: set `/strategy/converged=1`, `convergence_reason="user"` on current `status_{N}.h5` (no new file)
8. **Accumulate**: per-depth misfit history

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down. Aggregation uses `Aggregate.aggregate_misfits` from `shared/aggregate/`. Grid refinement and operator prompting come from `shared/grid/`.

## Tool Stack

- Julia (`HDF5.jl`, `Statistics.jl`, `LinearAlgebra.jl`)
- NaN-aware aggregation (masked misfits → NaN, skip in sums)
- Grid refinement logic (halve steps, center on best)

## Misfit Aggregation Hierarchy

```
phase → trial          (XCorr: one value per phase per trial)
channel → trial        (Polarity: one P-polarity value per channel per trial)
```

Each module aggregates independently, then weighted sum across modules. PSR misfits are optional — if absent from `/misfits/`, zeros are used.

## What It Does NOT Do

- Does NOT compute misfits (that's `forward.cpp`)
- Does NOT generate trials (that's `preprocess.jl`)
- Does NOT modify `database.h5`
