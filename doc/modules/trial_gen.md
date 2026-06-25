# Module: Grid (Trial Generation + Refinement)

**Location**: `shared/grid/` (Julia package `Grid`)

## Sub-modules

- `trial_gen.jl` — expands strategy grid into trial table (Cartesian product)
- `grid_refinement.jl` — computes next iteration's search grid from current results

Both are included and re-exported from `Grid.jl`.

## Purpose

Generate trials from strategy parameters (grid expansion). Cartesian product of varying axes.

## Used By

- `scripts/preprocess.jl` — trial generation (each loop iteration)
- `scripts/assess.jl` — grid refinement + operator prompt (each loop iteration)

### Trial Generation

**Input**:

Strategy from `status_{N}.h5`:

- `strike0`, `dstrike`, `nstrike` (SDR grid)
- `dip0`, `ddip`, `ndip` (SDR grid)
- `rake0`, `drake`, `nrake` (SDR grid)
- `depth_indices` (indices into `database.h5/config/depth_vals`)
- `freq_indices` (frequency band indices)

**Output**:

`/trials` group in `status_{N}.h5`:

- `strike[N]`, `dip[N]`, `rake[N]` (degrees)
- `depth[N]` (km, actual depth values)
- `depth_idx[N]` (indices into database)
- `freq_idx[N]` (frequency band indices)

Where `N = max(nstrike,1) × max(ndip,1) × max(nrake,1) × max(len(depth_indices),1) × max(len(freq_indices),1)`.

### Grid Refinement

Computes next iteration's grid parameters from current best trial.

**Input**: strategy + per-trial aggregated misfits from `assess.jl`

| Parameter                  | Source                     | Rule                                                       |
|----------------------------|----------------------------|------------------------------------------------------------|
| `strike0`, `dip0`, `rake0` | Current best trial SDR     | New grid **start** values (not center)                     |
| `dstrike`, `ddip`, `drake` | Current step sizes         | Halved: `new_step = current_step / 2`                      |
| `nstrike`, `ndip`, `nrake` | Fixed                      | Always `[3, 3, 3]` (3 values per SDR axis)                 |
| `depth_indices`            | `depth_misfit_accumulated` | Indices of depths within 20% of best depth misfit          |
| `freq_indices`             | `freq_misfit_curve`        | Indices of frequencies within 20% of best frequency misfit |

**Refinement factor**: fixed at 0.5 (half step sizes each iteration).

**Grid size**: fixed at 3 per SDR axis. Total trials = `3 × 3 × 3 × N_depths × N_freqs`.

#### Edge Cases

- Depth subset empty → use best depth index only (`N_depths = 1`)
- Frequency subset empty → use best freq index only (`N_freqs = 1`)
- First iteration (`status_0.h5`) → initial strategy set by config, no refinement

### Operator Prompt

`grid_refinement.jl` includes `prompt_operator()` which displays current best result and asks the operator whether to continue (`y`/`N`).

- **y** → creates `status_{N+1}.h5` with `/strategy/converged=0`. Driver loops to preprocess for next iteration.
- **N** (any other) → sets `/strategy/converged=1`, `convergence_reason="user"` on the **current** `status_{N}.h5` (no new file created). Driver detects converged=1 and breaks to output.

### Output (Refinement)

Updated strategy for `status_{N+1}.h5` (on continue):

- `strike0`, `dstrike`, `nstrike`
- `dip0`, `ddip`, `ndip`
- `rake0`, `drake`, `nrake`
- `depth_indices`
- `freq_indices`
- `best_sdr`: `[strike, dip, rake]` from current best trial
- `best_misfit`: total misfit of current best trial
- `freq_accumulated`: best SDR per frequency band, for uncertainty reporting
- `freq_misfit_curve`: misfit values used for frequency subset selection
- `depth_misfit_accumulated`: best misfit per depth, for depth subset selection

## Rules

- Axis with `n=0` → not varying, contributes 1 value (uses `var0` only)
- Empty `depth_indices` → no depth variation, use `best_depth_index` from strategy
- Empty `freq_indices` → no frequency variation, single frequency band
- Trial order: deterministic (same Cartesian product order every time)

## Testing Strategy

- Verify total trial count matches product of axis sizes
- Verify trial values match expected grid positions
- Grid center matches best trial from current iteration
- Step sizes halved correctly
- Grid size: always 3×3×3 SDR
- Depth/freq subsets contain best indices
- Edge case: single value on all axes (1 trial)
- Edge case: all axes varying
