# Data Interface: HDF5 Schema

## Dimension Symbols

| Symbol | Description | Typical Value |
|----------------------|--------------------------------------------------|-----------------|
| `N_stations` | Unique physical stations (no channel dimension) | 6–30 |
| `N_channels` | Unique (station, channel) pairs | 10–90 |
| `N_phases_P` | P-phase entries (one per station per channel) | 10–90 |
| `N_phases_S` | S-phase entries (one per station per channel) | 10–90 |
| `N_phases` | Total phase entries = N_phases_P + N_phases_S | 20–180 |
| `N_depths` | Depth levels for Green's functions | 10–40 |
| `N_bands` | Frequency band combinations | configurable |
| `N_durations` | Gaussian STF duration candidates (σ) | configurable |
| `N_modules` | Active misfit modules (counted by instance) | 2–3 |
| `N_samples_raw` | Raw waveform samples per channel before trimming | input-dependent |
| `N_samples` | Trimmed waveform samples per phase | 200–20000 |
| `N_polarity_samples` | Polarity window samples | 50–200 |

All datasets use `Float64` unless noted. Scalars stored as scalar datasets.

Phase key convention: `{network}.{station}.{channel}.{phase_type}`.
Channel ID convention: `{station_id}.{channel}` (e.g. `NET.ST1.Z`).

### Index Convention

All array/vector indices throughout the schema are **1-based** (Julia convention).
Values correspond directly to Julia array indexing. Zero is not a valid index.

| Field | Group | Description |
|--------------------|------------------------|---------------------------------------------------------|
| `depth_indices` | `/strategy` | Depth indices to search (1..N_depths) |
| `freq_indices` | `/strategy` | Frequency band indices to search (1..N_bands) |
| `duration_indices` | `/strategy` | STF duration indices to search (1..N_durations) |
| `band_low` | `/config/{ModuleName}` | Low-cut index into `/paraspace/frequency` (per-module) |
| `band_high` | `/config/{ModuleName}` | High-cut index into `/paraspace/frequency` (per-module) |
| `station_idx` | `/{ModuleName}` | Station table index (1..N_stations) |
| `depth_idx` | `/trials` | GF depth index per trial (1..N_depths) |
| `freq_idx` | `/trials` | Frequency band index per trial (1..N_bands) |
| `duration_idx` | `/trials` | STF duration index per trial (1..N_durations) |

This applies to all HDF5 files (`database.h5`, `status_N.h5`, `output.h5`).

______________________________________________________________________

## `database.h5` — Static Data (Written by input.jl)

### `/paraspace`

Expanded float arrays for all parameter-space dimensions. Values are the actual
parameter values (not indices). Downstream stages reference these via integer
indices stored in `/strategy` and `/trials` in `status_{N}.h5`.

| Dataset | Type | Shape | Description |
|-------------|---------|-----------------|-----------------------------------------------------------------------------------|
| `strike` | Float64 | `[N_strike]` | All strike values from grid expansion (deg) |
| `dip` | Float64 | `[N_dip]` | All dip values from grid expansion (deg) |
| `rake` | Float64 | `[N_rake]` | All rake values from grid expansion (deg) |
| `depth` | Float64 | `[N_depths]` | All depth levels (km) |
| `frequency` | Float64 | `[N_freq]` | Discrete frequency values (Hz). Low/high cuts of each band index into this array. |
| `duration` | Float64 | `[N_durations]` | Gaussian STF σ candidates (s) |

Dimension sizes are determined by the initial grid configuration. `/strategy`
in `status_{N}.h5` provides integer indices into these arrays.

### `/config`

Algorithm metadata and module-specific parameters. No float parameter-space
values or integer indices — those live in `/paraspace` and `/strategy` respectively.

| Dataset | Type | Shape | Description |
|------------------|--------|---------------|----------------------------------------------------------------|
| `misfit_modules` | String | `[N_modules]` | Active objectives (baseline: XcorrP/S, LagP/S, Psr, PolarityP) |
Per-module settings in sub-groups, named after each module instance as listed
in `misfit_modules`. Present only when the module is active:

- **`/config/{ModuleName}/`**: XCorr and Lag base instances have
  `max_lag_periods`, `filter_order`, `trim`, `band_low`, `band_high`. Expression
  objectives carry `bases` and `primitives` instead of preprocessing parameters.

Each module group also carries misfit-decomposition metadata (see
`doc/misfit-decomposition.md`):

| Dataset | Type | Shape | Description |
|---------------|--------|--------|------------------------------------------------------------------------------------------------------------|
| `operator` | String | scalar | Pipeline operator name (`"Xcorr"`/`"Expression"`/legacy composer) |
| `output` | String | scalar | Selected output field (`"cc_max"`/`"best_lag"`/...) |
| `is_composed` | Int8 | scalar | 0=Level 1 base, 1=Level 2 composed |
| `phase` | String | scalar | Phase type (`"P"`/`"S"`) — Level 1 only |
| `channel` | String | scalar | Channel filter (`""`=none, `"Z"`/`"N"`/`"E"`) - Level 1 only |
| `bases` | String | `[k]` | Base misfit names — Level 2 only |
| `primitives` | String | `[k]` | DSL primitive requirements (`max_cc`/`lag_cc`/`energy`/`rms`/`amp_scale`/`sign_scale`) — `Expression` only |

DSL 目标函数注册后写入 `/config/objectives/{ObjectiveName}/`。表达式以递归节点组存储：

| Dataset/group | Type | Description |
| `kind` | String | `"input"`、`"waveform"`、`"literal"` 或 `"call"` |
| `name` | String | 输入节点名称，仅 `input` 节点存在 |
| `role` / `phase` | String | `observed`/`synthetic` 与 `P`/`S`，仅 `waveform` 节点存在 |
| `band` / `window` | Float64 | 两元数组，仅 `waveform` 节点存在 |
| `channel` / `filter_order` | String / Int | 通道过滤与滤波器阶数，仅 `waveform` 节点存在 |
| `value` | Number | 常量值，仅 `literal` 节点存在 |
| `op` | String | 算子名称，仅 `call` 节点存在 |
| `args/{1..N}` | Group | 按位置编号的子表达式 |
| `kwargs/{name}` | scalar | 算子关键字参数，例如 `maxlag` |

DSL 编译器将根 XCorr/lag 目标降低为基础计算，将其余合法组合降低为通用 `Expression`；`misfit_modules` 和每模块配置是编译产物，供后续阶段直接使用。

| Dataset | Type | Shape | Description |
|-------------|-------|-------------|------------------------------------------------------|
| `band_low` | Int32 | `[N_bands]` | Low-cut indices into `/paraspace/frequency` (XCorr) |
| `band_high` | Int32 | `[N_bands]` | High-cut indices into `/paraspace/frequency` (XCorr) |

### `/event`

| Dataset | Type | Shape | Description |
|--------------|---------|--------|------------------------|
| `longitude` | Float64 | scalar | Event longitude (deg) |
| `latitude` | Float64 | scalar | Event latitude (deg) |
| `depth` | Float64 | scalar | Event depth (km) |
| `magnitude` | Float64 | scalar | Event magnitude |
| `origintime` | String | scalar | Origin time (ISO 8601) |

### `/station`

Flat arrays indexed by `N_stations` (one row per unique **physical station**,
without channel dimension).

| Dataset | Type | Shape | Description |
|--------------|---------|----------------|---------------------------------------|
| `id` | String | `[N_stations]` | Station identifier (`NET.ST1`) |
| `network` | String | `[N_stations]` | Network code |
| `station` | String | `[N_stations]` | Station name |
| `latitude` | Float64 | `[N_stations]` | Station latitude (deg) |
| `longitude` | Float64 | `[N_stations]` | Station longitude (deg) |
| `elevation` | Float64 | `[N_stations]` | Station elevation (m) |
| `dt` | Float64 | `[N_stations]` | Sampling interval (s) |
| `begin_time` | String | `[N_stations]` | Recording start time (ISO 8601) |
| `distance` | Float64 | `[N_stations]` | Epicentral distance (km) |
| `azimuth` | Float64 | `[N_stations]` | Event-to-station azimuth (deg) |
| `P_time` | String | `[N_stations]` | P-wave arrival time (ISO 8601) |
| `S_time` | String | `[N_stations]` | S-wave arrival time (ISO 8601) |
| `P_polarity` | Int8 | `[N_stations]` | P-wave first-motion polarity (1/-1/0) |

### `/channel`

One dataset per channel-station pair. Dataset name: `{station_id}.{channel}`
(e.g. `NET.ST1.Z`). Shape `[N_samples_raw]`.

### `/gf`

Green's functions, grouped by **depth index**. One dataset per channel-station
pair per depth.

Depth group names are **1-based indices into `/paraspace/depth`** (matching
`/trials/depth_idx`); physical depth values live only in `/paraspace/depth`, so
group names never carry float-formatted depth strings.

```
/gf/{idx}/{station_id}.{channel}    Float64[N_samples_raw × 6]   GF matrix (time × MT components)
```

### `/{ModuleName}` — Per-Misfit-Module Data

Each active misfit module instance gets its own group, named after the module
instance as listed in `misfit_modules`. Level 1 XCorr/Lag instances own data
groups; composed `Expression` objectives only reference their bases and primitive requirements.

The internal structure follows a general schema:

**Per-group metadata (group root):**

| Dataset | Type | Shape | Description |
|---------------|--------|---------------|----------------------------------------|
| `channel_id` | String | `[N_entries]` | Channel identifier per entry |
| `station_idx` | Int32 | `[N_entries]` | Index into `/station` tables (1-based) |

**`/{ModuleName}/obs/{band}/`**

Observation data per frequency band. `band` is the 1-indexed band number.

| Dataset | Type | Shape | Description |
|-------------|---------|--------------------------|-----------------------------------------------------------|
| `obs` | Float64 | `[N_entries, N_samples]` | Preprocessed observed data (XCorr: fixed obs window) |
| `obs_norm2` | Float64 | `[N_entries]` | Energy of each trace (XCorr modules only) |
| `obs_psr` | Float64 | `[N_entries]` | `log10(rms_P / rms_S)` amplitude ratio (PSR modules only) |

**`/{ModuleName}/gf/{idx}/{band}/{duration_idx}/`**

Green's function data per depth, frequency band, and STF duration index.

| Dataset | Type | Shape | Description |
|----------|---------|-----------------------------|--------------------------------------------------------------------------------------|
| `gf` | Float64 | `[N_entries, 6, N_samples]` | Preprocessed Green's functions |
| `amp_P` | Float64 | `[N_entries, 6, 6]` | GFᵀ·GF within P window (PSR only) |
| `amp_S` | Float64 | `[N_entries, 6, 6]` | GFᵀ·GF within S window (PSR only) |
| `synamp` | Float64 | `[N_entries, 6, 6]` | Single-window GF auto-correlation — legacy (pre per-lag refactor), no longer written |

The exact shape dimensions depend on the module type:

- For XCorr instances: `N_entries = N_phases_{P,S}` (number of phase entries)
- For Polarity: `N_entries = N_channels` (number of channels)
- For PSR: `N_entries = N_stations` (one entry per station with both P and S picks)

Depth group names follow `/gf/{idx}` (1-based index into `/paraspace/depth`),
consistent with `/trials/depth_idx`.

XCorr reductions also carry the duration level:

```
/{ModuleName}/synamp_lag/{depth_idx}/{band}/{duration_idx}
/{ModuleName}/dot_obs_gf_lag/{band}/{duration_idx}
```

The Layer 0 `/preprocess` and `/gf_preprocessed` debug persistence is not part
of the current schema.

> Active PSR and polarity are Expression objectives over XCorr data. Legacy
> standalone `obs_psr`/`amp_P`/`amp_S` and Polarity preprocessing fields are not used.

______________________________________________________________________

## `status_{N}.h5` — Per-Iteration Workflow File

One file per iteration, built incrementally by pipeline stages.

### `/strategy`

Current-iteration search grid definition. SDR axes are expanded inline
(start + k·step, `n` values) to build the trial space; `depth_indices`,
`freq_indices`, and `duration_indices` select subsets of `/paraspace/depth`,
`/paraspace/frequency`, and `/paraspace/duration`.
Plus the iteration counter.

| Dataset | Type | Shape | Description |
|--------------------|---------|--------|--------------------------------------------------------|
| `strike0` | Float64 | scalar | Strike grid start (deg) |
| `dstrike` | Float64 | scalar | Strike step (deg) |
| `nstrike` | Int32 | scalar | Strike count (72 = full space 5°, wraps 0..355) |
| `dip0` | Float64 | scalar | Dip grid start (deg) |
| `ddip` | Float64 | scalar | Dip step (deg) |
| `ndip` | Int32 | scalar | Dip count (19 = full space 5°) |
| `rake0` | Float64 | scalar | Rake grid start (deg) |
| `drake` | Float64 | scalar | Rake step (deg) |
| `nrake` | Int32 | scalar | Rake count (37 = full space 5°) |
| `depth_indices` | Int32 | `[n]` | Indices into `/paraspace/depth` |
| `freq_indices` | Int32 | `[n]` | Indices into `/paraspace/frequency` bands (1..N_bands) |
| `duration_indices` | Int32 | `[n]` | Indices into `/paraspace/duration` (1..N_durations) |
| `iteration` | Int32 | scalar | Iteration number |

The full-space 5° grid (initial iteration) is the single source of truth
`IO.DEFAULT_GRID` / `Search.default_grid()`. Current `assess.jl` converges after
the first iteration and does not write `status_{N+1}.h5`.

### `/trials`

| Dataset | Type | Shape | Description |
|----------------|-------|--------------|------------------------------------------------------|
| `strike_idx` | Int32 | `[N_trials]` | Strike axis index into `/paraspace/strike` (1-based) |
| `dip_idx` | Int32 | `[N_trials]` | Dip axis index into `/paraspace/dip` (1-based) |
| `rake_idx` | Int32 | `[N_trials]` | Rake axis index into `/paraspace/rake` (1-based) |
| `depth_idx` | Int32 | `[N_trials]` | Depth index into `/paraspace/depth` (1-based) |
| `freq_idx` | Int32 | `[N_trials]` | Frequency band index |
| `duration_idx` | Int32 | `[N_trials]` | STF duration index into `/paraspace/duration` |
| `N_trials` | Int32 | scalar | Trial count |

Trials carry **indices only** — physical values (strike/dip/rake/depth/duration)
are not stored per trial. They live exclusively in the `/paraspace`
axis arrays and are resolved on demand (forward MT conversion, `output.jl`
best-trial/uncertainty).

### `/intermediates`

Raw kernel intermediate products (written by C++ forward, consumed by
Julia `assess.jl`). Grouped by canonical key `{Operator}{Phase}[_{channel}]`
(e.g. `XcorrP`, `XcorrS`). Multiple misfit instances sharing the same
(operator, phase, channel) share one intermediate group.

| Group | Dataset | Type | Shape | Description |
|--------------|--------------|---------|-------------------------|-------------------------------------------------|
| `Xcorr{P,S}` | `cc_max` | Float64 | `[N_phases × N_trials]` | signed max normalized CC value |
| `Xcorr{P,S}` | `best_lag` | Int32 | `[N_phases × N_trials]` | best-lag offset (samples, relative to maxlag) |
| `Xcorr{P,S}` | `syn_energy` | Float64 | `[N_phases × N_trials]` | center-lag `mᵀGᵀGm`; written when PSR is active |
| `XcorrP` | `amp_scale` | Float64 | `[N_phases × N_trials]` | half peak-to-peak synthetic amplitude |
| `XcorrP` | `sign_scale` | Int8 | `[N_phases × N_trials]` | sign of the earlier min/max extremum |

`assess.jl` transforms these into final misfits via Output extractors
(Level 1) and Composers (Level 2). See `doc/misfit-decomposition.md`.

### `/misfits`

Raw per-module misfits (unweighted, unaggregated). One dataset per module instance.

### `/aggregate`

Assess first averages entries per trial, min-max maps each objective to `[0, 1]`,
then averages all objectives with equal weight. `/aggregate/normalized/{ModuleName}`
stores each normalized trial vector; `/aggregate/total` stores the final trial score.

| Dataset | Type | Shape | Description |
|----------------|---------|--------------------------|--------------------------|
| `{ModuleName}` | Float64 | `[N_entries x N_trials]` | per-module misfit matrix |

______________________________________________________________________

## `output.h5` — Final Results

(Unchanged — see previous schema version.)

### `/solution`

| Dataset | Type | Shape | Description |
|-----------------|---------|--------|--------------------------------------------------------------------|
| `strike` | Float64 | scalar | Best-fit strike (deg) |
| `dip` | Float64 | scalar | Best-fit dip (deg) |
| `rake` | Float64 | scalar | Best-fit rake (deg) |
| `depth` | Float64 | scalar | Best-fit depth (km) |
| `duration` | Float64 | scalar | Best-fit Gaussian STF σ (s) |
| `duration_idx` | Float64 | scalar | Best-fit index into `/paraspace/duration` |
| `moment_tensor` | Float64 | `[6]` | [Mxx, Myy, Mzz, Mxy, Mxz, Myz] |
| `misfit` | Float64 | scalar | Current equal-weight normalized aggregate across active objectives |

### `/uncertainty`

| Dataset | Type | Shape | Description |
|--------------------------|---------|--------------------------------|-------------------------|
| `strike_std` | Float64 | scalar | Strike uncertainty |
| `dip_std` | Float64 | scalar | Dip uncertainty |
| `rake_std` | Float64 | scalar | Rake uncertainty |
| `depth_range` | Float64 | `[2]` | Depth bounds [min, max] |
| `freq_test_misfit_curve` | Float64 | `[N_bands, N_freq_test_mechs]` | Misfit vs frequency |

### `/per_phase`

Phase-level misfit breakdown for the best trial.

| Dataset | Type | Shape | Description |
|---------------------|---------|--------------------------|-----------------------------------|
| `phase_id` | String | `[N_phases]` | Phase identifiers |
| `station_id` | String | `[N_phases]` | Station identifiers |
| `phase_type` | String | `[N_phases]` | "P" or "S" |
| `misfit_per_module` | Float64 | `[N_modules x N_phases]` | Final misfit per module per phase |
| `selected` | Int32 | `[N_phases]` | Phase selected in final solution |
| `cross_correlation` | Float64 | `[N_phases]` | Best XCorr per phase |

### `/per_station_summary`

| Dataset | Type | Shape | Description |
|--------------------------|---------|----------------|----------------------------------|
| `station_id` | String | `[N_stations]` | Station identifiers |
| `n_phases` | Int32 | `[N_stations]` | Number of phases per station |
| `mean_cross_correlation` | Float64 | `[N_stations]` | Mean XCorr across station phases |
| `misfit_total` | Float64 | `[N_stations]` | Aggregate misfit per station |

### `/summary`

| Dataset | Type | Shape | Description |
|----------------------|--------|--------|------------------------|
| `total_iterations` | Int32 | scalar | Total iterations |
| `total_trials` | Int32 | scalar | Total trials evaluated |
| `convergence_reason` | String | scalar | Why pipeline stopped |
