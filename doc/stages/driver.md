# Stage: `driver.sh` — Pipeline Orchestration

## Role

Orchestrates the 5-stage pipeline. Stateless — all state lives in HDF5 files. Sequentially runs stages; `assess.jl` signals continue (exit 0) or converged (exit 10).

## Inputs

| Source | Purpose |
|-----------------|-----------------------------------------------------------------------------|
| `config.jl` | Bootstrap config (passed to `input.jl` only; always `<data-dir>/config.jl`) |
| `database.h5` | Preprocessed data (path known to all stages) |
| `status_{N}.h5` | Iteration snapshots (discovered by file inspection) |

## Outputs

| Output | Producer |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `database.h5` | `input.jl` (once) |
| `status_{N}.h5` | `input.jl` (strategy), `preprocess.jl` (trials), `forward.cpp` (misfits), `assess.jl` (convergence on break; creates `status_{N+1}.h5` on continue) |
| `output.h5` | `output.jl` |

## Responsibilities

1. **Stage invocation** — call `input.jl`, `preprocess.jl`, `forward.cpp`, `assess.jl`, `output.jl` in order
1. **Loop control** — run preprocess→forward→assess repeatedly until assess exits with code 10 (converged)
1. **File-level checks** — check `database.h5` existence (triggers input once). All state detection is delegated to assess.jl.
1. **Error handling** — stop on failure, report error to stderr

## Pipeline Stage Detection

| Condition | Action |
|--------------------------|------------------------------------------------------------------|
| No `database.h5` | Run `input.jl` (once, with `config.jl`) |
| `database.h5` exists | Loop: `preprocess.jl` → `forward.cpp` → `assess.jl` indefinitely |
| `assess.jl` exit code 10 | Break loop → run `output.jl` |

All HDF5 group-level state detection (trials/misfits existence, converged flag) is handled by `assess.jl` internally. The driver only checks `database.h5` file existence.

## Tool Stack

- Bash (built-in file tests, loops, string parsing)
- Julia runner (scripts use `include()` for shared packages; helpers use `julia --project=shared/io`)
- Compiled `forward` binary

## CLI

```
bash driver.sh --data-dir <dir>
```

- `--data-dir <dir>` (required): data directory; holds `config.jl`, `database.h5`, `output.h5`; contains `status/` subdir with `status_{N}.h5` files

## Key Decisions

- **Bootstrapping**: Config passed only to `input.jl`. Subsequent runs read strategy from `status_{N}.h5`.
- **Resume**: Re-running driver picks up from current state. `database.h5` exists → skips input. Assess.jl checks iteration state internally.
- **Convergence**: `assess.jl` prompts operator; on continue exit 0 → creates `status_{N+1}.h5` with refined strategy (converged=0). On break exit 10 → sets `/strategy/converged=1` on the **current** `status_{N}.h5` — no new file is created. Driver tests exit code to break to output.

## What It Does NOT Do

- Does NOT compute anything — pure orchestration
- Does NOT generate or modify HDF5 data — only checks whether `database.h5` exists
