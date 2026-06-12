# Module: Trial Generation

## Purpose

Generate trials from strategy parameters (grid expansion). Cartesian product of varying axes.

## Used By

- `preprocess.jl` (all runs — each loop iteration)

## Input

Strategy from `status_{N}.h5`:
- `strike0`, `dstrike`, `nstrike` (SDR grid)
- `dip0`, `ddip`, `ndip` (SDR grid)
- `rake0`, `drake`, `nrake` (SDR grid)
- `depth_indices` (indices into `database.h5/config/depth_vals`)
- `freq_indices` (frequency band indices)

## Output

`/trials` group in `status_{N}.h5`:
- `strike[N]`, `dip[N]`, `rake[N]` (degrees)
- `depth[N]` (km, actual depth values)
- `depth_idx[N]` (indices into database)
- `freq_idx[N]` (frequency band indices)

Where `N = nstrike × ndip × nrake × len(depth_indices) × len(freq_indices)`.

## Rules

- Axis with `n=0` → not varying, use `var0` only
- Empty `depth_indices` → no depth variation, use `best_depth_index` from strategy
- Empty `freq_indices` → no frequency variation, single frequency band
- Trial order: deterministic (same Cartesian product order every time)

## Testing Strategy

- Verify total trial count matches product of axis sizes
- Verify trial values match expected grid positions
- Edge case: single value on all axes (1 trial)
- Edge case: all axes varying