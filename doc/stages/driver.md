# Stage: `driver.sh` — Pipeline Orchestration

## Role

Orchestrates the 5-stage pipeline. Stateless — all state lives in HDF5 files.

**Current state:** stage 1 (input) only. The pipeline loop (preprocess→forward→assess) and output stage are defined but unreachable — `exit 0` at end of input stage. Restoring full pipeline is pending.

## Inputs

| Source | Purpose |
|---------------|-----------------------------------------------------------------------------|
| `config.jl` | Bootstrap config (passed to `input.jl` only; always `<data-dir>/config.jl`) |
| `database.h5` | Preprocessed data (produced by `input.jl`) |

## Outputs

| Output | Producer |
|---------------------------------|----------------------------|
| `database.h5` | `input.jl` (once) |
| `status_0.h5` → `status/<N>.h5` | `input.jl` (strategy only) |

## Responsibilities (current)

1. **Stage 1 invocation** — call `input.jl` with config file path
1. **File-level checks** — check data directory and config file existence
1. **Status file location** — move `status_0.h5` into `status/` subdirectory
1. **Error handling** — stop on failure, report error to stderr

## Pipeline Stage Detection (current)

| Condition | Action |
|------------------------------------------------|---------------------------------------|
| No `--data-dir` / missing dir / missing config | Exit with error |
| `database.h5` exists | Warn and continue (always runs input) |
| All checks pass | Run `input.jl` once, then exit 0 |

## Tool Stack

- Bash (built-in file tests, string parsing, tee logging)
- Julia runner (`julia --project=root`)

## CLI

```
bash driver.sh --data-dir <dir>
```

- `--data-dir <dir>` (required): data directory; must contain `config.jl`
- Log files: `driver.log`, per-stage logs (`input.log`, etc.) written to data dir

## Key Decisions

- **Bootstrapping**: Config passed only to `input.jl`. All config values written to `database.h5`; subsequent stages read from HDF5.
- **Status files**: `status_0.h5` written by `input.jl` to data dir root, then moved into `status/` subdirectory.
- **Convergence**: Assess.jl exit codes `0` (continue) / `10` (converged) — not yet wired in driver.
- **Logging**: Color-aware when stdout is a terminal; always tee to `driver.log`.

## Pending (defined but not wired)

| Feature | Code exists | Driver wiring |
|--------------------------------|-----------------|--------------------|
| preprocess→forward→assess loop | Lines 124-146 | Behind `exit 0` |
| Output stage | Lines 148-152 | Behind `exit 0` |
| Assess exit code 10 | In `assess.jl` | Not read by driver |
| Resume / skip input | Not implemented | — |

## What It Does NOT Do

- Does NOT compute anything — pure orchestration
- Does NOT modify HDF5 data directly (only moves files)
