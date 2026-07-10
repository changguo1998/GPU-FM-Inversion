# Data Interface: HDF5 Schema

## Dimension Symbols

| Symbol | Description | Typical Value |
|----------------------|--------------------------------------------------|-----------------|
| `N_stations` | Unique stations | 10–30 |
| `N_channels` | Unique (station, channel) pairs | 10–90 |
| `N_phases_P` | P-phase entries (one per station per channel) | 10–90 |
| `N_phases_S` | S-phase entries (one per station per channel) | 10–90 |
| `N_phases` | Total phase entries = N_phases_P + N_phases_S | 20–180 |
| `N_depths` | Depth levels for Green's functions | 10–40 |
| `N_bands` | Frequency band combinations | configurable |
| `N_modules` | Active misfit modules | 2–3 |
| `N_samples_raw` | Raw waveform samples per channel before trimming | input-dependent |
| `N_samples` | Trimmed waveform samples per phase | 200–20000 |
| `N_polarity_samples` | Polarity window samples | 50–200 |

All datasets use `Float64` unless noted. Scalars stored as scalar datasets.

Phase key convention: `{network}.{station}.{channel}.{phase_type}`.
Channel ID convention: `{station_id}.{channel}` (e.g. `NET.ST1.Z`).

______________________________________________________________________

## `database.h5` — Static Data (Written by input.jl)

### `/config`

| Dataset | Type | Shape | Description |
|--------------------|---------|---------------|-----------------------------------------|
| `misfit_modules` | String | `[N_modules]` | Active modules: `"XCorr"`, `"Polarity"` |
| `depth_vals` | Float64 | `[N_depths]` | All depth levels |
| `n_bands` | Int32 | scalar | Number of frequency bands |
| `freq_bands_low` | Float64 | `[N_bands]` | Low-cut corner frequencies (Hz) |
| `freq_bands_high` | Float64 | `[N_bands]` | High-cut corner frequencies (Hz) |
| `minimum_stations` | Int32 | scalar | Minimum stations required |

Per-module settings in sub-groups (present only when module is in `misfit_modules`):

- **`/config/xcorr/`**: `maxlag_factor` (scalar), `filter_order` (Int32), `P_trim` [2], `S_trim` [2], `select_threshold` (scalar), `deselect_threshold` (scalar)
- **`/config/polarity/`**: `trim` [2]

### `/event`

| Dataset | Type | Shape | Description |
|--------------|---------|--------|------------------------|
| `longitude` | Float64 | scalar | Event longitude (deg) |
| `latitude` | Float64 | scalar | Event latitude (deg) |
| `depth` | Float64 | scalar | Event depth (km) |
| `magnitude` | Float64 | scalar | Event magnitude |
| `origintime` | String | scalar | Origin time (ISO 8601) |

### `/station`

Flat arrays indexed by `N_stations` (one row per unique station).

| Dataset | Type | Shape | Description |
|--------------|---------|----------------|---------------------------------------|
| `id` | String | `[N_stations]` | Station identifier (`NET.ST1`) |
| `network` | String | `[N_stations]` | Network code |
| `station` | String | `[N_stations]` | Station name |
| `channel` | String | `[N_stations]` | Channel code (`Z`, `N`, `E`) |
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

One dataset per channel-station pair. Dataset name: `{station_id}.{channel}` (e.g. `NET.ST1.Z`).

| Dataset | Type | Shape | Description |
|--------------------------|---------|-------------------|-------------------------|
| `{station_id}.{channel}` | Float64 | `[N_samples_raw]` | Raw continuous waveform |

### `/gf`

Green's functions, grouped by depth. One dataset per channel-station pair per depth.

```
/gf/{depth}/{station_id}.{channel}    Float64[N_samples_raw × 6]   GF matrix (time × MT components)
```

### `/xcorr`

XCorr module preprocessed data. Observed data stored once (no depth dimension); GF data stored per depth.

**`/xcorr/obs/{phasetype}-{band}/`**

| Dataset | Type | Shape | Description |
|-------------|---------|-------------------------|---------------------------------------|
| `obs` | Float64 | `[N_phases, N_samples]` | Filtered + trimmed observed waveforms |
| `obs_norm2` | Float64 | `[N_phases]` | Energy of each obs trace |

Where `phasetype` ∈ {P, S}, `band` is the 1-indexed band number.

**`/xcorr/gf/{depth}/{phasetype}-{band}/`**

| Dataset | Type | Shape | Description |
|----------|---------|----------------------------|-----------------------------------------|
| `gf` | Float64 | `[N_phases, 6, N_samples]` | Filtered + trimmed Green's functions |
| `synamp` | Float64 | `[N_phases, 6, 6]` | GF auto-correlation (gf^T gf) per phase |

### `/polarity`

Polarity module preprocessed data.

**`/polarity/obs/`**

| Dataset | Type | Shape | Description |
|-----------|---------|----------------|-------------------------------------------------|
| `obs_pol` | Float64 | `[N_channels]` | Observed polarity values (-1.0, 0.0, +1.0, NaN) |

**`/polarity/gf/{depth}/`**

| Dataset | Type | Shape | Description |
|----------|---------|---------------------------------------|---------------------------|
| `gf_pol` | Float64 | `[N_channels, 6, N_polarity_samples]` | GF within polarity window |

### `/index`

Flat arrays indexed by `N_phases` (one row per phase entry).

| Dataset | Type | Shape | Description |
|--------------------|---------|-------------------------|------------------------------------------|
| `phase_ids` | String | `[N_phases]` | Phase identifiers (`NET.ST1.Z.P`) |
| `phase_type` | String | `[N_phases]` | `"P"` or `"S"` |
| `station_idx` | Int32 | `[N_phases]` | Index into `/station` tables (1-based) |
| `distance` | Float64 | `[N_phases]` | Epicentral distance (km) per phase |
| `azimuth` | Float64 | `[N_phases]` | Event-to-station azimuth (deg) per phase |
| `greens_depth_idx` | Int32 | `[N_phases × N_depths]` | GF depth index per phase per depth |

______________________________________________________________________

## `status_{N}.h5` — Per-Iteration Workflow File

One file per iteration, built incrementally by pipeline stages.

### `/strategy`

Grid axes: `n > 0` means axis varies, generating `n` values as `var0 + i * dvar` for i = 0..n-1.

| Dataset | Type | Shape | Description |
|-----------------|---------|--------|--------------------------------|
| `strike0` | Float64 | scalar | Strike start (deg) |
| `dstrike` | Float64 | scalar | Strike step (deg) |
| `nstrike` | Int32 | scalar | Strike value count (0 = fixed) |
| `dip0` | Float64 | scalar | Dip start (deg) |
| `ddip` | Float64 | scalar | Dip step (deg) |
| `ndip` | Int32 | scalar | Dip value count (0 = fixed) |
| `rake0` | Float64 | scalar | Rake start (deg) |
| `drake` | Float64 | scalar | Rake step (deg) |
| `nrake` | Int32 | scalar | Rake value count (0 = fixed) |
| `depth_indices` | Int32 | `[n]` | Depth indices to search |
| `freq_indices` | Int32 | `[n]` | Freq band indices to search |
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

### `/misfits`

Raw per-module misfits (unweighted, unaggregated).

| Dataset | Type | Shape | Level |
|------------|---------|---------------------------|---------|
| `xcorr` | Float64 | `[N_phases × N_trials]` | phase |
| `polarity` | Float64 | `[N_channels × N_trials]` | channel |
| `psr` | Float64 | `[N_channels × N_trials]` | channel |

______________________________________________________________________

## `output.h5` — Final Results

(Unchanged from previous schema.)

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
| `misfit_per_module` | Float64 | `[N_modules × N_phases]` | Final misfit per module per phase |
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
