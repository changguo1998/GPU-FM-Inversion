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
|-----------------|-------------|-----------------------------------------------|
| `depth_indices` | `/strategy` | Depth indices to search (1..N_depths) |
| `freq_indices` | `/strategy` | Frequency band indices to search (1..N_bands) |
|| `band_low` | `/config/{ModuleName}` | Low-cut index into `/paraspace/frequency` (per-module) |
|| `band_high` | `/config/{ModuleName}` | High-cut index into `/paraspace/frequency` (per-module) |
| `station_idx` | `/{ModuleName}` | Station table index (1..N_stations) |
| `depth_idx` | `/trials` | GF depth index per trial (1..N_depths) |
| `freq_idx` | `/trials` | Frequency band index per trial (1..N_bands) |
| `depth_idx` | `Grid.TrialResult` | Best depth index (1..N_depths) |
| `freq_idx` | `Grid.TrialResult` | Best frequency index (1..N_bands) |

This applies to all HDF5 files (`database.h5`, `status_N.h5`, `output.h5`).

______________________________________________________________________

## `database.h5` — Static Data (Written by input.jl)

### `/paraspace`

Expanded float arrays for all parameter-space dimensions. Values are the actual
parameter values (not indices). Downstream stages reference these via integer
indices stored in `/strategy` and `/trials` in `status_{N}.h5`.

| Dataset | Type | Shape | Description |
|-------------|---------|--------------|-----------------------------------------------------------------------------------|
| `strike` | Float64 | `[N_strike]` | All strike values from grid expansion (deg) |
| `dip` | Float64 | `[N_dip]` | All dip values from grid expansion (deg) |
| `rake` | Float64 | `[N_rake]` | All rake values from grid expansion (deg) |
| `depth` | Float64 | `[N_depths]` | All depth levels (km) |
| `frequency` | Float64 | `[N_freq]` | Discrete frequency values (Hz). Low/high cuts of each band index into this array. |

Dimension sizes are determined by the initial grid configuration. `/strategy`
in `status_{N}.h5` provides integer indices into these arrays.

### `/config`

Algorithm metadata and module-specific parameters. No float parameter-space
values or integer indices — those live in `/paraspace` and `/strategy` respectively.

| Dataset | Type | Shape | Description |
|------------------|--------|---------------|------------------------------------------------|
| `misfit_modules` | String | `[N_modules]` | Active module instance names (e.g. `"XcorrP"`) |
Per-module settings in sub-groups, named after each module instance as listed
in `misfit_modules`. Present only when the module is active:

- **`/config/{ModuleName}/`**: Parameters depend on the module type. XCorr
  instances have `maxlag_factor`, `filter_order`, `trim`, `select_threshold`,
  `deselect_threshold`, `band_low`, `band_high`. Polarity has `trim`.

Each module group also carries misfit-decomposition metadata (see
`doc/misfit-decomposition.md`):

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `operator` | String | scalar | Operator name (`"Xcorr"`/`"Polarity"`/`"StdDev"`) |
| `output` | String | scalar | Selected output field (`"cc_max"`/`"best_lag"`/...) |
| `is_composed` | Int8 | scalar | 0=Level 1 base, 1=Level 2 composed |
| `phase` | String | scalar | Phase type (`"P"`/`"S"`) — Level 1 only |
| `channel` | String | scalar | Channel filter (`""`=none, `"H"`/`"V"`) — Level 1 only |
| `bases` | String | `[k]` | Base misfit names — Level 2 only |

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

Green's functions, grouped by depth. One dataset per channel-station pair per depth.

```
/gf/{depth}/{station_id}.{channel}    Float64[N_samples_raw × 6]   GF matrix (time × MT components)
```

### `/{ModuleName}` — Per-Misfit-Module Data

Each active misfit module instance gets its own group, named after the module
instance as listed in `misfit_modules` (e.g. `XcorrP`, `XcorrS`, `Polarity`).

The internal structure follows a general schema:

**Per-group metadata (group root):**

| Dataset | Type | Shape | Description |
|---------------|--------|---------------|----------------------------------------|
| `channel_id` | String | `[N_entries]` | Channel identifier per entry |
| `station_idx` | Int32 | `[N_entries]` | Index into `/station` tables (1-based) |

**`/{ModuleName}/obs/{band}/`**

Observation data per frequency band. `band` is the 1-indexed band number.

| Dataset | Type | Shape | Description |
|-------------|---------|--------------------------|-------------------------------------------|
| `obs` | Float64 | `[N_entries, N_samples]` | Preprocessed observed data |
| `obs_norm2` | Float64 | `[N_entries]` | Energy of each trace (XCorr modules only) |

**`/{ModuleName}/gf/{depth}/{band}/`**

Green's function data per depth and frequency band.

| Dataset | Type | Shape | Description |
|----------|---------|-----------------------------|------------------------------------------|
| `gf` | Float64 | `[N_entries, 6, N_samples]` | Preprocessed Green's functions |
| `synamp` | Float64 | `[N_entries, 6, 6]` | GF auto-correlation (XCorr modules only) |

The exact shape dimensions depend on the module type:

- For XCorr instances: `N_entries = N_phases_{P,S}` (number of phase entries)
- For Polarity: `N_entries = N_channels` (number of channels)

______________________________________________________________________

## `status_{N}.h5` — Per-Iteration Workflow File

One file per iteration, built incrementally by pipeline stages.

### `/strategy`

Integer indices into `/paraspace` arrays, plus iteration counter.
Trial generation reads the expanded float values from `/paraspace`
and selects subsets by these indices.

| Dataset | Type | Shape | Description |
|-----------------|-------|--------|--------------------------------------------------------|
| `depth_indices` | Int32 | `[n]` | Indices into `/paraspace/depth` |
| `freq_indices` | Int32 | `[n]` | Indices into `/paraspace/frequency` bands (1..N_bands) |
| `iteration` | Int32 | scalar | Iteration number |

### `/trials`

| Dataset | Type | Shape | Description |
|-------------|---------|--------------|----------------------|
| `strike` | Float64 | `[N_trials]` | Strike angles (deg) |
| `dip` | Float64 | `[N_trials]` | Dip angles (deg) |
| `rake` | Float64 | `[N_trials]` | Rake angles (deg) |
| `depth` | Float64 | `[N_trials]` | Depth (km) |
| `depth_idx` | Int32 | `[N_trials]` | GF depth index |
| `freq_idx` | Int32 | `[N_trials]` | Frequency band index |
| `N_trials` | Int32 | scalar | Trial count |

### `/intermediates`

Raw kernel intermediate products (written by C++ forward, consumed by
Julia `assess.jl`). Grouped by canonical key `{Operator}{Phase}[_{channel}]`
(e.g. `XcorrP`, `XcorrS`, `PolarityP`). Multiple misfit instances sharing
the same (operator, phase, channel) share one intermediate group.

| Group | Dataset | Type | Shape | Description |
|-------|---------|------|-------|-------------|
| `Xcorr{P,S}` | `cc_max` | Float64 | `[N_phases × N_trials]` | max normalized CC value |
| `Xcorr{P,S}` | `best_lag` | Int32 | `[N_phases × N_trials]` | best-lag offset (samples, relative to maxlag) |
| `Polarity{P}` | `syn_sign` | Int8 | `[N_stations × N_trials]` | synthetic polarity sign (-1/0/1) |
| `Polarity{P}` | `dot_value` | Float64 | `[N_stations × N_trials]` | raw dot product (confidence) |

`assess.jl` transforms these into final misfits via Output extractors
(Level 1) and Composers (Level 2). See `doc/misfit-decomposition.md`.

### `/misfits`

Raw per-module misfits (unweighted, unaggregated). One dataset per module instance.

| Dataset | Type | Shape | Description |
|----------------|---------|--------------------------|--------------------------|
| `{ModuleName}` | Float64 | `[N_entries x N_trials]` | per-module misfit matrix |

______________________________________________________________________

## `output.h5` — Final Results

(Unchanged — see previous schema version.)

### `/solution`

| Dataset | Type | Shape | Description |
|-----------------|---------|--------|--------------------------------|
| `strike` | Float64 | scalar | Best-fit strike (deg) |
| `dip` | Float64 | scalar | Best-fit dip (deg) |
| `rake` | Float64 | scalar | Best-fit rake (deg) |
| `depth` | Float64 | scalar | Best-fit depth (km) |
| `moment_tensor` | Float64 | `[6]` | [Mxx, Myy, Mzz, Mxy, Mxz, Myz] |
| `misfit` | Float64 | scalar | Final weighted misfit |

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
