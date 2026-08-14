# Stage: `scripts/assess.jl` — Misfit Extraction, Composition, Convergence

## Role

Runs once per iteration (preprocess → forward → **assess**). Reads the raw
intermediates from `status_N.h5:/intermediates/`, transforms them into misfit
matrices (Level 1 extractors), aggregates composed modules (Level 2 composers),
and writes the final per-module misfits to `status_N.h5:/misfits/`. Also emits
the iteration convergence decision consumed by `driver.sh`.

## Usage

```bash
julia scripts/assess.jl <database.h5> <status_N.h5>
```

When `DATA_DIR` is set in the environment, an empty `.decision.txt` is written
into it (converged). Exit code 0 on success.

## Inputs

| File | Used for |
|---------------|------------------------------------------------------------------------|
| `database.h5` | `/config` (module list + params), `/station`, `/{Module}/station_idx` |
| `status_N.h5` | `/intermediates/{key}/` (read), `/trials` (count), `/misfits/` (write) |

## Processing

1. **Read context**: config modules, stations, trial count.

1. **Level 1 — extract**: for each non-composed module, map its
   `(operator, output)` to an extractor in `Aggregate.EXTRACTORS`:

   | (operator, output) | Extraction |
   |------------------------------------------------------|--------------------------------------|
   | `(:Xcorr, :cc_max)` | `1 .− cc_max` (normalized-CC misfit) |
   | `(:Xcorr, :best_lag)` | `best_lag · dt` (absolute shift, s) |
   | `(:Polarity, :syn_sign)` / `(:Polarity, :dot_value)` | deferred |

   Intermediate rows are transposed to `[entries × trials]` (the C++ writes
   C-order `[N_phases × N_trials]`, which HDF5.jl reads reversed).

1. **Level 2 — compose**: composed modules (`is_composed == 1`) select their
   base misfits and run `Aggregate.COMPOSERS[op]` (e.g. `StdDev`), resolved
   topologically over `bases`.

1. **Write `/misfits/`**: one dataset per module, shape `[N_entries × N_trials]`,
   replacing any previous value (idempotent).

1. **Convergence decision**: current implementation converges on the first
   iteration (writes an empty `.decision.txt`). Weighted aggregation and grid
   refinement (multi-iteration loop) are the next planned feature — see
   `doc/roadmap.md` Phase 2.

## Outputs

### `status_N.h5:/misfits/{ModuleName}`

| Dataset | Type | Shape | Description |
|----------|---------|--------------------------|--------------------------|
| `XcorrS` | Float64 | `[N_entries × N_trials]` | per-module misfit matrix |

### `{DATA_DIR}/.decision.txt`

Empty file = converged (driver exits the loop). Non-empty = continue.

## Notes

- `read_intermediate` uses the (N_trials, N_entries) heuristic on the read
  shape to normalize the C++ C-order storage into `[entries × trials]`.
- XCorr-only mode: only `XcorrS` is registered in sample configs, so Level 2
  composition is exercised only by the `Aggregate` package tests.
- assess does NOT modify `/strategy` or `/trials` (grid refinement is pending).
- Baseline (2026-08-14, after MT-formula fix in `tests/synthetic_data.jl`):
  best trial = true source (30, 60, 90) @ 10 km, XCorrS mean misfit ≈ 0.1602
  — see `AGENTS.md`.
