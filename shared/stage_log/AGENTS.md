# AGENTS.md — StageLog module (`shared/stage_log/src/StageLog.jl`)

## Role

Per-stage logging with prefix labels. Each pipeline stage sets up its own logger that prefixes messages with the stage name (`[input]`, `[assess]`, etc.) and writes to both stdout/stderr and a named log file.

Used by: all stage scripts (`input.jl`, `preprocess.jl`, `assess.jl`, `output.jl`).

## Exports

| Function        | Signature                                    | Role                                                                                            |
|-----------------|----------------------------------------------|-------------------------------------------------------------------------------------------------|
| `setup_logger!` | `(prefix::String, filename::AbstractString)` | Create `StageLogger` with prefix, set as global logger, register `atexit` handler to close file |

## Type

`StageLogger <: AbstractLogger` — custom Julia `Logging` framework logger.

| Field    | Type       | Description                                                   |
|----------|------------|---------------------------------------------------------------|
| `prefix` | `String`   | Stage label, e.g. `"input"` — rendered as `[input]` in output |
| `io`     | `IOStream` | Log file handle                                               |

## Logging behavior

- `@info` → `"[prefix] message"` to stdout + log file
- `@warn` → `"WARN: message"` to stderr + log file
- `@error` → `"ERROR: message"` to stderr + log file
- `min_enabled_level` returns `BelowMinLevel` (all messages pass)
- `shouldlog` always returns true

## Log file locations

| Stage      | Log file                      |
|------------|-------------------------------|
| input      | `<data_dir>/input.log`        |
| preprocess | `<status_dir>/preprocess.log` |
| assess     | `<status_dir>/assess.log`     |
| output     | `<status_dir>/output.log`     |

## Coding conventions

- One logger per stage — created once at stage entry, set as global logger.
- `atexit` handler ensures log file is closed on normal exit or error.
- No multi-threading considerations — stages are single-threaded Julia processes.
- No log rotation — files are created with `"w"` mode (overwrite each run).