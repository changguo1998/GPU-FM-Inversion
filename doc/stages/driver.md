# Stage: `driver.sh` — Pipeline Orchestration

## Role

Orchestrates the 4-stage pipeline loop. Stateless — all state lives in HDF5 files. Determines next action by inspecting file state and `/strategy/converged` flag.

## Inputs

| Source | Purpose |
|--------|---------|
| `raw.h5` | Input data (passed to setup on first run) |
| `config.toml` | Bootstrap config (passed to setup on first run only) |
| `database.h5` | Preprocessed data (path known to all stages) |
| `status_{N}.h5` | Iteration snapshots (discovered by file inspection) |

## Outputs

| Output | Producer |
|--------|----------|
| `database.h5` | `setup.jl` (first run) |
| `status_{N}.h5` | `setup.jl`, `forward.cpp`, `assess.jl` |
| `output.h5` | `export.jl` |

## Responsibilities

1. **State detection** — inspect HDF5 files and group presence to determine next stage
2. **Stage invocation** — call `setup.jl`, `forward.cpp`, `assess.jl`, `export.jl` in order
3. **Loop control** — detect converged flag, loop or break to export
4. **Error handling** — stop on failure, report error to stderr

## Pipeline Stage Detection

| File State | Action |
|-----------|--------|
| No `database.h5` | Run `setup.jl` (first run, with `config.toml`) |
| `status_{N}.h5` exists, no `/trials` | Run `setup.jl` (write trials from strategy) |
| `status_{N}.h5` exists, has `/trials`, no `/misfits` | Run `forward.cpp` |
| `status_{N}.h5` exists, has `/misfits` | Run `assess.jl` |
| `status_{N+1}.h5` exists, `/strategy/converged == 1` | Run `export.jl` |

## Tool Stack

- Bash (built-in file tests, loops, string parsing)
- Julia runner (`julia --project=<stage_dir>`)
- Compiled `forward` binary
- HDF5 introspection via `julia -e "using HDF5; ..."`

## Key Decisions

- **Bootstrapping**: `config.toml` passed only to first-run `setup.jl`. Subsequent runs read strategy from `status_{N}.h5`.
- **Resume**: Re-running driver picks up from current state based on file/group existence.
- **Convergence**: `assess.jl` prompts operator; on break, sets `/strategy/converged=1` in `status_{N+1}.h5`. Driver checks this flag to break to export.

## What It Does NOT Do

- Does NOT compute anything — pure orchestration
- Does NOT generate or modify HDF5 data — only reads `/strategy/converged` for loop control and checks file/group existence
