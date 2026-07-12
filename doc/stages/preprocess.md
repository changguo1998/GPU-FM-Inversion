# Stage: `scripts/preprocess.jl` — Trial Generation

## Role

Runs once per iteration in the main pipeline loop (preprocess → forward → assess).
Reads the current strategy from `status_N.h5`, generates the full set of trial
parameters (strike × dip × rake × depth × freq) as a Cartesian product, and
writes them into `/trials` in the same `status_N.h5`. The forward stage then
reads `/trials` to compute misfits for each trial.

Also reads `depth_vals` from `database.h5` to map depth indices to physical
depth values (km).

## Usage

```bash
FM_DATA_DIR=/path/to/data julia scripts/preprocess.jl
```

No CLI arguments. Files are located via `ENV["FM_DATA_DIR"]`:

| File | Path (relative to `FM_DATA_DIR`) | Access |
|---------------|----------------------------------|---------------------------------------|
| `database.h5` | `database.h5` | Read (`/config/depth_vals`) |
| latest status | `status/status_N.h5` | Read (`/strategy`), Write (`/trials`) |

The driver.sh exports `FM_DATA_DIR` before invoking each stage.

## Inputs

### `status_N.h5` → `/strategy`

| Dataset | Type | Shape | Description |
|-----------------|---------|--------|-------------------------------------|
| `strike0` | Float64 | scalar | Strike start (deg) |
| `dstrike` | Float64 | scalar | Strike step (deg) |
| `nstrike` | Int32 | scalar | Number of strike values (0 = fixed) |
| `dip0` | Float64 | scalar | Dip start (deg) |
| `ddip` | Float64 | scalar | Dip step (deg) |
| `ndip` | Int32 | scalar | Number of dip values (0 = fixed) |
| `rake0` | Float64 | scalar | Rake start (deg) |
| `drake` | Float64 | scalar | Rake step (deg) |
| `nrake` | Int32 | scalar | Number of rake values (0 = fixed) |
| `depth_indices` | Int32 | `[n]` | Depth indices to search |
| `freq_indices` | Int32 | `[n]` | Freq band indices to search |
| `iteration` | Int32 | scalar | Iteration number |

### `database.h5` → `/config`

| Dataset | Type | Shape | Description |
|--------------|---------|--------------|-----------------------|
| `depth_vals` | Float64 | `[N_depths]` | All depth levels (km) |

Only `depth_vals` is needed — it maps 1-based `depth_indices` to km values
for the `/trials/depth` dataset (see Index Convention in `doc/schema.md`).

## Outputs

### `status_N.h5` → `/trials`

| Dataset | Type | Shape | Description |
|-------------|---------|--------------|--------------------------------|
| `strike` | Float64 | `[N_trials]` | Strike angles (deg) |
| `dip` | Float64 | `[N_trials]` | Dip angles (deg) |
| `rake` | Float64 | `[N_trials]` | Rake angles (deg) |
| `depth` | Float64 | `[N_trials]` | Depth (km) |
| `depth_idx` | Int32 | `[N_trials]` | GF depth index (1-based) |
| `freq_idx` | Int32 | `[N_trials]` | Frequency band index (1-based) |
| `N_trials` | Int32 | scalar | Trial count |

## Responsibilities

1. **Find latest status file**: scan `status/` directory for highest `N` in
   `status_N.h5`. The file already exists from either `input.jl` (iteration 0)
   or the previous `assess.jl` (iteration N+1).
1. **Read strategy**: load `/strategy` group via `IO.read_strategy()`.
1. **Read depth_vals**: load `/config/depth_vals` from `database.h5` via
   `IO.read_config()`.
1. **Convert to GridStrategy**: extract strategy fields into
   `Grid.GridStrategy`, including `depth_indices`.
1. **Generate trials**: call `Grid.generate_trials(strategy, depth_vals)`.
1. **Write trials**: replace `/trials` group via `IO.write_trials()`.

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down when executed.

- Shared modules via `using IO, Grid, StageLog`
- Logger prefix: `"preprocess"`
- Log file: `{FM_DATA_DIR}/preprocess.log`

## Dependencies

- `IO.jl` — read_strategy, read_config, write_trials, find_latest_status
- `Grid.jl` — GridStrategy, generate_trials
- `StageLog.jl` — setup_logger!

## What It Does NOT Do

- Does NOT modify `/strategy` (assess.jl writes the next strategy).
- Does NOT read or write `/misfits` (forward stage writes misfits).
- Does NOT compute misfits or apply weights.
- Does NOT prompt the operator (assess.jl handles interaction).
- Does NOT create `status_N.h5` — the file must already exist with `/strategy`
  from `input.jl` or `assess.jl`.

## Design Notes (from review)

### `GridStrategy` must include `depth_indices`

`Grid.generate_trials()` iterates over `strategy.depth_indices` to generate
depth-axis trials. The `GridStrategy` struct in `trial_gen.jl` is missing
this field — documented in AGENTS.md but not implemented. Will be added.

### `Grid.TrialSet` to be removed, `IO.TrialSet` used exclusively

`Grid.TrialSet` and `IO.TrialSet` are identical structs (same fields, same
types). The duplicate will be removed; `Grid.generate_trials()` will return
`H5IO.TrialSet` (the IO module's type, accessed via Grid's `H5IO` alias).
This eliminates a type-conversion step in `preprocess.jl`.

### Environment variable, not CLI args

Stage scripts take no CLI arguments. The driver.sh exports `FM_DATA_DIR` to
locate data files. This keeps the driver.sh in control of directory layout
and keeps stage scripts simple.

### 1-based index convention

All indices (`depth_indices`, `freq_indices`, `station_idx`, `depth_idx`, `freq_idx`) are 1-based. See Index Convention in `doc/schema.md` for the full table.
