# Stage: `scripts/preprocess.jl` — Trial Generation

## Role

Runs once per iteration in the main pipeline loop (preprocess → forward → assess).
Reads the current strategy from `status_N.h5`, generates the full set of trial
parameters as a Cartesian product of per-axis **indices**
(strike × dip × rake × depth × freq — all 1-based indices into `/paraspace`),
and writes them into `/trials` in the same `status_N.h5`. The forward stage
then reads `/trials` and resolves physical values from `/paraspace`.

Trials carry **indices only** — physical values (angles in deg, depth in km)
are not stored per trial; they live exclusively in `database.h5:/paraspace`
and are resolved on demand (forward MT conversion, output best-trial reporting).

## Usage

```bash
DATA_DIR=/path/to/data julia scripts/preprocess.jl
```

No CLI arguments. Files are located via `ENV["DATA_DIR"]` (exported by driver.sh):

| File | Path (relative to `DATA_DIR`) | Access |
|---------------|-------------------------------|---------------------------------------|
| latest status | `status/status_N.h5` | Read (`/strategy`), Write (`/trials`) |

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

No database.h5 read: physical axis values are not needed at this stage.

## Outputs

### `status_N.h5` → `/trials`

| Dataset | Type | Shape | Description |
|--------------|-------|--------------|------------------------------------------------------|
| `strike_idx` | Int32 | `[N_trials]` | Strike axis index into `/paraspace/strike` (1-based) |
| `dip_idx` | Int32 | `[N_trials]` | Dip axis index into `/paraspace/dip` (1-based) |
| `rake_idx` | Int32 | `[N_trials]` | Rake axis index into `/paraspace/rake` (1-based) |
| `depth_idx` | Int32 | `[N_trials]` | Depth index into `/paraspace/depth` (1-based) |
| `freq_idx` | Int32 | `[N_trials]` | Frequency band index (1-based) |
| `N_trials` | Int32 | scalar | Trial count |

## Responsibilities

1. **Find latest status file**: scan `status/` directory for highest `N` in
   `status_N.h5`. The file already exists from either `input.jl` (iteration 0)
   or the previous `assess.jl` (iteration N+1).
1. **Read strategy**: load `/strategy` group via `IO.read_strategy()`.
1. **Generate trials**: call `Grid.generate_trials(strategy)` — consumes the
   full `IO.Strategy` (SDR grid dims + depth/freq indices), produces per-axis
   1-based index vectors (no physical values).
1. **Write trials**: replace `/trials` group via `IO.write_trials()`.

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down when executed.

- Shared modules via `using IO, Grid, StageLog`
- Logger prefix: `"preprocess"`
- Log file: `{DATA_DIR}/preprocess.log`

## Dependencies

- `IO.jl` — read_strategy, write_trials, find_latest_status
- `Grid.jl` — generate_trials
- `StageLog.jl` — setup_logger!

## What It Does NOT Do

- Does NOT modify `/strategy` (assess.jl writes the next strategy).
- Does NOT read or write `/misfits` (forward stage writes misfits).
- Does NOT read `/paraspace` or `database.h5` at all.
- Does NOT compute misfits or apply weights.
- Does NOT prompt the operator (assess.jl handles interaction).
- Does NOT create `status_N.h5` — the file must already exist with `/strategy`
  from `input.jl` or `assess.jl`.

## Design Notes (from review)

- `GridStrategy`/`Grid.TrialSet` duplicates were removed (2026-08-07): `Grid.generate_trials`
  consumes the full `IO.Strategy` and returns `IO.TrialSet` directly — no conversion step.
- Trials are all-indices (2026-08-10): `TrialSet` carries `strike_idx/dip_idx/rake_idx/ depth_idx/freq_idx` — no physical values. `generate_trials(strategy)` takes no `depth_vals`.
- Stage scripts take no CLI arguments; driver.sh exports `DATA_DIR` to locate data files.
- All indices (`depth_indices`, `freq_indices`, `station_idx`, `strike_idx`, `dip_idx`,
  `rake_idx`, `depth_idx`, `freq_idx`) are 1-based. See Index Convention in `doc/schema.md`.

### 1-based index convention
