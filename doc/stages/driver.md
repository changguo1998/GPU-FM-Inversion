# Stage: `driver.sh` — Pipeline Orchestration

## Role

Orchestrates the 5-stage pipeline. Stateless — all state lives in HDF5 files. Determines next action by inspecting file state and `/strategy/converged` flag.

## Inputs

| Source | Purpose |
|--------|---------|
| `raw.h5` | Input data (passed to `input.jl` once) |
| `config.toml` | Bootstrap config (passed to `input.jl` only; default path is `<data-dir>/config.toml`, override with `--config`) |
| `database.h5` | Preprocessed data (path known to all stages) |
| `status_{N}.h5` | Iteration snapshots (discovered by file inspection) |

## Outputs

| Output | Producer |
|--------|----------|
| `database.h5` | `input.jl` (once) |
| `status_{N}.h5` | `input.jl` (strategy), `preprocess.jl` (trials), `forward.cpp` (misfits), `assess.jl` (convergence on break; creates `status_{N+1}.h5` on continue) |
| `output.h5` | `output.jl` |

## Responsibilities

1. **State detection** — inspect HDF5 files and group presence to determine next stage
2. **Stage invocation** — call `input.jl`, `preprocess.jl`, `forward.cpp`, `assess.jl`, `output.jl` in order
3. **Loop control** — detect converged flag, loop or break to output
4. **Error handling** — stop on failure, report error to stderr

## Pipeline Stage Detection

| File State | Action |
|-----------|--------|
| No `database.h5` | Run `input.jl` (once, with `config.toml`) |
| `status_{N}.h5` exists, no `/trials` | Run `preprocess.jl` (generate trials from strategy) |
| `status_{N}.h5` exists, has `/trials`, no `/misfits` | Run `forward.cpp` |
| `status_{N}.h5` exists, has `/misfits` | Run `assess.jl` |
| `status_{N}.h5` exists, `/strategy/converged == 1` | Run `output.jl` |

## Tool Stack

- Bash (built-in file tests, loops, string parsing)
- Julia runner (scripts use `include()` for shared packages; helpers use `julia --project=shared/io`)
- Compiled `forward` binary
- HDF5 introspection via `julia -e "using HDF5; ..."`

## CLI

```
bash driver.sh [--data-dir <dir>] [--config <path>] [--dry-run] [--synthetic]
```

- `--data-dir <dir>`: directory for `raw.h5`, `database.h5`, status files (default: `.`)
- `--config <path>`: TOML config file path (default: `<data-dir>/config.toml`)
- `--dry-run`: print stages without executing
- `--synthetic`: generate synthetic data via input.jl before running pipeline

## Key Decisions

- **Bootstrapping**: Config passed only to `input.jl`. Subsequent runs read strategy from `status_{N}.h5`.
- **Resume**: Re-running driver picks up from current state based on file/group existence.
- **Convergence**: `assess.jl` prompts operator; on continue, creates `status_{N+1}.h5` with refined strategy (converged=0). On break, sets `/strategy/converged=1` on the **current** `status_{N}.h5` — no new file is created. Driver checks the latest status file for the converged flag to break to output.

## What It Does NOT Do

- Does NOT compute anything — pure orchestration
- Does NOT generate or modify HDF5 data — only reads `/strategy/converged` for loop control and checks file/group existence
