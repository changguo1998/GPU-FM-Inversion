# Stage: `assess.jl` — Weighting, Aggregation & Strategy Update

## Role

Reads raw misfits from `status_{N}.h5`, applies module weights and phase masks, aggregates into per-trial scores, refines the search grid, and prompts the operator whether to continue or break. Writes updated strategy to `status_{N+1}.h5`.

**Owns all strategy decisions**: module weights, phase selection, grid refinement. `forward.cpp` never applies weights.

## Inputs

| Source | Description |
|--------|-------------|
| `status_{N}.h5` | Reads `/trials`, `/misfits`, `/strategy` |
| `database.h5` | Reads `/config`, `/index` (for reference) |

## Outputs

| Source | Description |
|--------|-------------|
| `status_{N+1}.h5` | Writes `/strategy` (refined grid, masks, weights, best result, converged flag) |

## Responsibilities

1. **Apply per-module masks**: each module has its own phase/station mask
2. **Weight and aggregate**: sum masked misfits per module, apply module weights, combine → per-trial total
3. **Find best trial**: argmin of aggregated misfit
4. **Refine grid**: compute next search grid from current best SDR (halve step sizes, 3×3×3 fixed)
5. **Prompt operator**: display current results, ask "Continue? [y/N]"
6. **Write strategy**: save refined grid to `status_{N+1}.h5`; set `/strategy/converged=1` if operator chose break
7. **Accumulate**: frequency test results, per-depth misfit history

## Tool Stack

- Julia (`HDF5.jl`, `Statistics.jl`, `LinearAlgebra.jl`)
- NaN-aware aggregation (masked misfits → NaN, skip in sums)
- Grid refinement logic (halve steps, center on best)

## Misfit Aggregation Hierarchy

```
phase → trial          (XCorr: one value per phase per trial)
station → trial        (Polarity: one P-polarity value per station per trial)
station → trial        (PSR: one value per station per trial)
```

Each module aggregates independently, then weighted sum across modules.

## What It Does NOT Do

- Does NOT compute misfits (that's `forward.cpp`)
- Does NOT generate trials (that's `setup.jl`)
- Does NOT modify `database.h5`
