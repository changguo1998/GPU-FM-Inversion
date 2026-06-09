# HDF5 Schema Design

## Common Types and Dimensions

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `N_stations` | Number of stations | 10–30 |
| `N_phases` | Total phase-station pairs (P+S per station) | 20–60 |
| `N_samples` | Waveform samples per phase (signal length) | 200–20000 |
| `N_polarity_samples` | Polarity window samples | 50–200 |
| `N_depths` | Depth levels for Green's functions | 10–40 |
| `N_frequencies` | Frequency band combinations | configurable |
| `N_modules` | Misfit modules (XCorr, Polarity, PSR, AbsShift, RelShift) | up to 5 |
| `N_trials` | Trials per iteration | 10–100000 |
| `N_components` | Moment tensor components (Mxx,Myy,Mzz,Mxy,Mxz,Myz) | 6 |
| `N_spatial` | Spatial components (E, N, Z) | 3 |

**Primary key convention**: each station-phase pair is identified by a composite key
`{network}.{station}.{component}.{phase_type}`.

All datasets use `Float64` unless noted. Scalars are `Float64` attributes, not datasets.

---

## 1. `raw.h5` — External Data (Static)

Written once by an import step. Contains only data from external sources, never modified.

### `/event` (group)

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `longitude` | Float64 | scalar | Event longitude (°) |
| `latitude` | Float64 | scalar | Event latitude (°) |
| `depth` | Float64 | scalar | Event depth (km) |
| `magnitude` | Float64 | scalar | Event magnitude |
| `origintime` | String | scalar | ISO 8601 datetime |

### `/phase_picks` (group)

Observational phase arrival times, keyed by station code.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `station_ids` | String | `[N_stations]` | Station identifiers `"NET.STA"` |
| `P_time` | String | `[N_stations]` | P-wave arrival ISO 8601 (empty string if none) |
| `S_time` | String | `[N_stations]` | S-wave arrival ISO 8601 (empty string if none) |
| `P_polarity` | Int8 | `[N_stations]` | Observed P polarity: -1, 0, +1, or -128 (not available) |

### `/stations` (group)

Station metadata, one row per station-channel.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `id` | String | `[N_phases]` | Full key `"NET.STA.COMP.TYPE"` |
| `network` | String | `[N_phases]` | Network code |
| `station` | String | `[N_phases]` | Station code |
| `component` | String | `[N_phases]` | Channel component (single char, e.g. "Z") |
| `latitude` | Float64 | `[N_phases]` | Station latitude (°) |
| `longitude` | Float64 | `[N_phases]` | Station longitude (°) |
| `elevation` | Float64 | `[N_phases]` | Station elevation (m) |
| `dt` | Float64 | `[N_phases]` | Original sampling interval (s) |
| `begin_time` | String | `[N_phases]` | ISO 8601 — original record start time |

### `/waveforms` (group)

Raw observed SAC traces. One dataset per station-phase pair for variable-length support.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `{phase_id}` | Float64 | `[N_samples_raw]` | Raw waveform for phase `id`, in velocity/acceleration |

---

## 2. `database.h5` — Preprocessed Data (Static)

Written once by `setup.jl`. All frequency bands and depth-dependent data precomputed upfront.

### `/config` (group)

Algorithm configuration, read from external TOML/YAML (see §5).

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `misfit_modules` | String | `[N_modules]` | Module names: `"XCorr"`, `"Polarity"`, `"PSR"`, ... |
| `module_weights` | Float64 | `[N_modules]` | Initial weights per module |
| `depth_vals` | Float64 | `[N_depths]` | All depth levels for GF + depth search |
| `freq_bands_low` | Float64 | `[N_frequencies]` | Low-cut corner frequencies (Hz) |
| `freq_bands_high` | Float64 | `[N_frequencies]` | High-cut corner frequencies (Hz) |
| `minimum_stations` | Int32 | scalar | Minimum stations required |
| `freq_test_max_iter` | Int32 | scalar | Frequency test maximum iterations |

Per-module settings live in sub-groups that exist **only when the module is in `misfit_modules`**. If a subgroup is absent, the module is not configured and will be skipped.

**`/config/xcorr/`** — present only when `"XCorr" ∈ misfit_modules`:

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `maxlag_factor` | Float64 | scalar | Max lag = factor / band_high |
| `filter_order` | Int32 | scalar | Butterworth filter order (default 4) |
| `P_trim` | Float64 | `[2]` | P-wave trim window relative to arrival, e.g. `[-2.0/band_high, 3.0/band_high]` |
| `S_trim` | Float64 | `[2]` | S-wave trim window, e.g. `[-4.0/band_high, 6.0/band_high]` |
| `select_threshold` | Float64 | scalar | Cross-correlation threshold to include channel |
| `deselect_threshold` | Float64 | scalar | Cross-correlation threshold to exclude channel |

**`/config/polarity/`** — present only when `"Polarity" ∈ misfit_modules`:

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `trim` | Float64 | `[2]` | Polarity window relative to arrival `[0, tsource]` |

### `/greens` (group)

Green's function waveforms per station per depth. 6 MT-component time series.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `{phase_id}/{depth_idx}` | Float64 | `[N_samples × 6]` | GF for phase `id` at depth index `depth_idx`, columns = [Mxx, Myy, Mzz, Mxy, Mxz, Myz] |

Per-GF attributes:
- `dt`: Float64 — time step
- `tp`: Float64 — P-wave travel time
- `ts`: Float64 — S-wave travel time
- `model`: String — velocity model name

### `/data` (group)

Precomputed waveform variants. Structure:

```
/data/{freq_idx}/{module}/{phase_id}/
```

For each frequency band `freq_idx`, each misfit module `module`, and each phase `phase_id`:

| Module | Datasets | Shape | Description |
|--------|----------|-------|-------------|
| XCorr | `obs` | `[N_samples]` | Filtered + trimmed observed waveform |
| | `gf` | `[N_samples × 6]` | Filtered + trimmed Green's function, same window |
| | `synamp` | `[6 × 6]` | GF auto-correlation matrix for amplitude normalization |
| Polarity | `gf_pol` | `[N_polarity_samples × 6]` | GF samples within polarity window |
| | `obs_pol` | Int8 | scalar — observed polarity (-1, 0, +1) per phase, extracted from raw.h5 |
| PSR | `amp_P` | `[6 × 6]` | P-wave amplitude covariance matrix |
| | `amp_S` | `[6 × 6]` | S-wave amplitude covariance matrix |
| | `obs_psr` | Float64 | scalar — observed log10(P/S) amplitude ratio |
| AbsShift | `obs` | `[3 × N_samples]` | Observed per spatial component (E,N,Z) |
| | `gf` | `[3 × N_samples × 6]` | GF per spatial component |
| RelShift | `obs` | `[3 × N_samples]` | Observed per spatial component (concatenated for correlation) |
| | `gf` | `[3 × N_samples × 6]` | GF per spatial component |

### `/index` (group)

Maps phase IDs to their data locations.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `phase_ids` | String | `[N_phases]` | All phase identifiers |
| `phase_type` | String | `[N_phases]` | `"P"` or `"S"` |
| `station_idx` | Int32 | `[N_phases]` | Index into station metadata |
| `distance` | Float64 | `[N_phases]` | Epicentral distance (km) — computed by setup.jl |
| `azimuth` | Float64 | `[N_phases]` | Event-to-station azimuth (°) — computed by setup.jl |
| `greens_depth_idx` | Int32 | `[N_phases × N_depths]` | Which GF dataset to use per phase per depth (-1 if not available) |

---

## 3. `status_{N}.h5` — Per-Iteration Snapshots

One file per iteration. `status_0.h5` is created by `setup.jl` on first run from `database.h5/config`. Subsequent files are created by `assess.jl` with the strategy for the next iteration; `setup.jl` then fills in trials. Each file is self-contained — strategy, trials, and misfits for a single iteration.

**Lifecycle:**

```
setup.jl  → status_0.h5 (/strategy + /trials)   [strategy from /config]
forward   → status_0.h5 (+ /misfits)
assess.jl → status_1.h5 (/strategy)              [strategy for next iteration]
setup.jl  → status_1.h5 (+ /trials)
forward   → status_1.h5 (+ /misfits)
assess.jl → status_2.h5 (/strategy)
...
```

### `/strategy` (group)

Strategy that generated this iteration's trials. Written by `setup.jl` (iteration 0) or `assess.jl` (subsequent iterations).

The search grid is defined per-axis: `n > 0` means that axis varies, generating `n` values as `var0 + i * dvar` for `i = 0 … n-1`. Depth and frequency use explicit index lists instead (indices may be non-uniform).

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike0` | Float64 | scalar | Grid start: strike center (°) |
| `dstrike` | Float64 | scalar | Strike step size (°) |
| `nstrike` | Int32 | scalar | Number of strike steps (0 = not varying) |
| `dip0` | Float64 | scalar | Grid start: dip center (°) |
| `ddip` | Float64 | scalar | Dip step size (°) |
| `ndip` | Int32 | scalar | Number of dip steps (0 = not varying) |
| `rake0` | Float64 | scalar | Grid start: rake center (°) |
| `drake` | Float64 | scalar | Rake step size (°) |
| `nrake` | Int32 | scalar | Number of rake steps (0 = not varying) |
| `depth_indices` | Int32 | `[n]` | Depth indices to search (missing = not varying) |
| `freq_indices` | Int32 | `[n]` | Frequency indices to search (missing = not varying) |
| `channel_mask` | Int32 | `[N_phases]` | Phase selection mask (1=active, 0=deselected) |
| `module_weights` | Float64 | `[N_modules]` | Current module weights (updated between stages) |
| `best_sdr` | Float64 | `[3]` | Current best (strike, dip, rake) |
| `best_depth_index` | Int32 | scalar | Index of current best depth into `/config/depth_vals` |
| `best_misfit` | Float64 | scalar | Current best weighted misfit |
| `iteration` | Int32 | scalar | Iteration number (matches file name) |

The pipeline stage is derived from which axes vary. Total trials: `nstrike × ndip × nrake × len(depth_indices) × len(freq_indices)`.

### `/trials` (group)

Written by `setup.jl`. One trial = one set of variable parameters. All datasets are `[N_trials]` — fixed size per iteration, no appending.

**Fixed parameters** (what the search varies):

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike` | Float64 | `[N_trials]` | Strike angles (°) for each trial |
| `dip` | Float64 | `[N_trials]` | Dip angles (°) |
| `rake` | Float64 | `[N_trials]` | Rake angles (°) |
| `depth` | Float64 | `[N_trials]` | Depth for each trial (km) — same for all when depth not varying |

**Data slice references** (indices into `database.h5`):

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `freq_idx` | Int32 | `[N_trials]` | Which preprocessed frequency band to use |
| `depth_idx` | Int32 | `[N_trials]` | Which Green's function depth slice to use |

`forward.cpp` uses `freq_idx` and `depth_idx` to select the correct data slices from `database.h5`, then computes misfits for each trial × phase.

### `/misfits` (group)

Written by `forward.cpp`. Raw per-module misfits — no weighting, no aggregation.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `data` | Float64 | `[N_modules × N_phases × N_trials]` | `data[m, p, t]` = raw misfit of module `m` for phase `p` in trial `t` |

`assess.jl` reads this and applies:
1. Channel selection mask → zero out deselected phases
2. Module weights → weighted sum per module
3. Module aggregation → total misfit per trial

---

## 4. `output.h5` — Final Results

Written once by `export.jl`.

### `/solution` (group)

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike` | Float64 | scalar | Best-fit strike (°) |
| `dip` | Float64 | scalar | Best-fit dip (°) |
| `rake` | Float64 | scalar | Best-fit rake (°) |
| `depth` | Float64 | scalar | Best-fit depth (km) |
| `moment_tensor` | Float64 | `[6]` | Corresponding MT [Mxx,Myy,Mzz,Mxy,Mxz,Myz] |
| `misfit` | Float64 | scalar | Final weighted misfit |

### `/uncertainty` (group)

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike_std` | Float64 | scalar | Strike uncertainty from frequency test |
| `dip_std` | Float64 | scalar | Dip uncertainty |
| `rake_std` | Float64 | scalar | Rake uncertainty |
| `depth_range` | Float64 | `[2]` | Depth uncertainty bounds `[min, max]` |
| `freq_test_misfit_curve` | Float64 | `[N_frequencies, N_freq_test_mechs]` | Misfit vs frequency vs mechanism (from frequency test iterations) |

### `/per_station` (group)

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `station_id` | String | `[N_phases]` | Station identifiers |
| `phase_type` | String | `[N_phases]` | `"P"` or `"S"` |
| `misfit_per_module` | Float64 | `[N_modules × N_phases]` | Final misfit per module per phase |
| `selected` | Int32 | `[N_phases]` | Whether this phase was selected in final solution |
| `cross_correlation` | Float64 | `[N_phases]` | Best XCorr value per phase (for QC) |

### `/summary` (group)

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `total_iterations` | Int32 | scalar | Total pipeline iterations |
| `total_trials` | Int32 | scalar | Total trials evaluated |
| `pipeline_stages_completed` | String | `[N_stages]` | List of stages that completed |
| `convergence_reason` | String | scalar | Why the pipeline stopped |