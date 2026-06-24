# Data Interface: HDF5 Schema

## Dimension Symbols

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `N_stations` | Stations | 10–30 |
| `N_channels` | Unique (station, component) pairs (one station may have 1–3 channels) | 10–90 |
| `N_phases` | Phase entries per channel + wave type (P/S) | 20–60 |
| `N_samples_raw` | Raw waveform samples per phase before filtering/trimming | input-dependent |
| `N_samples` | Waveform samples per phase | 200–20000 |
| `N_polarity_samples` | Polarity window samples | 50–200 |
| `N_depths` | Depth levels for Greens | 10–40 |
| `N_frequencies` | Frequency band combinations | configurable |
| `N_freq_test_mechs` | Mechanisms evaluated per frequency test | configurable |
| `N_modules` | Active misfit modules | 3 (XCorr, Polarity, PSR) |
| `N_trials` | Trials per iteration | 10–100000 |
| `N_components` | MT components | 6 |

All datasets use `Float64` unless noted. Scalars are stored as scalar datasets unless noted otherwise.

Phase key convention: `{network}.{station}.{component}.{phase_type}`.

---

## 1. `database.h5` — Preprocessed Data (Static)

### `/config`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `misfit_modules` | String | `[N_modules]` | Active modules: `"XCorr"`, `"Polarity"`, `"PSR"` (AbsShift, RelShift deferred; CAP cancelled) |
| `module_weights` | Float64 | `[N_modules]` | Initial module weights |
| `depth_vals` | Float64 | `[N_depths]` | All depth levels |
| `freq_bands_low` | Float64 | `[N_frequencies]` | Low-cut corner frequencies (Hz) |
| `freq_bands_high` | Float64 | `[N_frequencies]` | High-cut corner frequencies (Hz) |
| `minimum_stations` | Int32 | scalar | Minimum stations required |
| `freq_test_max_iter` | Int32 | scalar | Frequency test max iterations |

Per-module settings in sub-groups (present only when module is in `misfit_modules`):

**`/config/xcorr/`**: `maxlag_factor` (scalar), `filter_order` (Int32), `P_trim` [2], `S_trim` [2], `select_threshold` (scalar), `deselect_threshold` (scalar)

**`/config/polarity/`**: `trim` [2]

**`/config/psr/`**: no required module-specific parameters in v1; PSR uses global frequency bands and preprocessed P/S amplitude windows.

### `/greens`

`{phase_id}/{depth_idx}` → Float64 `[N_samples × 6]` — GF columns: [Mxx, Myy, Mzz, Mxy, Mxz, Myz]

Attributes: `dt`, `tp`, `ts`, `model`

### `/data`

Structure: `/data/{freq_idx}/{module}/{phase_id}/`

| Module | Datasets | Shape | Description | Status |
|--------|----------|-------|-------------|--------|
| XCorr | `obs` | `[N_samples]` | Filtered + trimmed observed waveform | active |
| | `gf` | `[N_samples × 6]` | Filtered + trimmed Green's function | |
| | `synamp` | `[6 × 6]` | GF auto-correlation matrix | |
| Polarity | `gf_pol` | `[N_polarity_samples × 6]` | GF within polarity window. **Only written for P-wave phase_ids.** | active |
| | `obs_pol` | Float64 | Observed polarity (-1.0, 0.0, +1.0, NaN = unavailable). **Only written for P-wave phase_ids.** | |
| PSR | `amp_P` | `[6 × 6]` | P-wave amplitude covariance matrix | active |
| | `amp_S` | `[6 × 6]` | S-wave amplitude covariance matrix | |
| | `obs_psr` | Float64 | Observed log10(P/S) ratio | |

**Note**: PSR data is stored per P/S phase-pair using key `"{P_phase_id}|{S_phase_id}"` (e.g., `"NET.STA.Z.P|NET.STA.Z.S"`), not under a single `phase_id`.
| AbsShift | `obs` | `[3 × N_samples]` | Observed per spatial component | **deferred** |
| | `gf` | `[3 × N_samples × 6]` | GF per spatial component | |
| RelShift | `obs` | `[3 × N_samples]` | Observed per spatial component | **deferred** |
| | `gf` | `[3 × N_samples × 6]` | GF per spatial component | |
| CAP | `obs` | `[3 × N_samples]` | Three-component observed | **cancelled** |
| | `gf` | `[3 × N_samples × 6]` | Three-component GF | |

### `/index`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `phase_ids` | String | `[N_phases]` | All phase identifiers |
| `phase_type` | String | `[N_phases]` | `"P"` or `"S"` |
| `station_idx` | Int32 | `[N_phases]` | Index into station metadata |
| `distance` | Float64 | `[N_phases]` | Epicentral distance (km) |
| `azimuth` | Float64 | `[N_phases]` | Event-to-station azimuth (°) |
| `greens_depth_idx` | Int32 | `[N_phases × N_depths]` | GF dataset index per phase per depth (-1 if unavailable) |

---

## 2. `status_{N}.h5` — Per-Iteration Workflow File

One file per iteration. Built up incrementally during each loop: starts with `/strategy` only from `input.jl`, then `/trials` from `preprocess.jl`, then `/misfits` from `forward.cpp`. Assess reads the completed file and either creates `status_{N+1}.h5` (continue) or sets `/strategy/converged=1` on the current file (break).

### `/strategy`

Grid axes: `n > 0` means axis varies, generating `n` values as `var0 + i * dvar` for i = 0..n-1 (start model).

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike0` | Float64 | scalar | Strike start (°) |
| `dstrike` | Float64 | scalar | Strike step (°) |
| `nstrike` | Int32 | scalar | Strike value count (0 = fixed) |
| `dip0` | Float64 | scalar | Dip start (°) |
| `ddip` | Float64 | scalar | Dip step (°) |
| `ndip` | Int32 | scalar | Dip value count (0 = fixed) |
| `rake0` | Float64 | scalar | Rake start (°) |
| `drake` | Float64 | scalar | Rake step (°) |
| `nrake` | Int32 | scalar | Rake value count (0 = fixed) |
| `depth_indices` | Int32 | `[n]` | Depth indices to search (missing = fixed) |
| `freq_indices` | Int32 | `[n]` | Freq band indices to search (missing = fixed) |
| `xcorr_phase_mask` | Int32 | `[N_phases]` | XCorr phase selection (1=active, 0=skip) |
| `polarity_channel_mask` | Int32 | `[N_channels]` | Polarity channel selection (1=active, 0=skip) |
| `psr_channel_mask` | Int32 | `[N_channels]` | PSR channel selection (1=active, 0=skip) |
| `module_weights` | Float64 | `[N_modules]` | Current module weights |
| `best_sdr` | Float64 | `[3]` | Best (strike, dip, rake) |
| `best_depth_index` | Int32 | scalar | Best depth index into `/config/depth_vals` |
| `best_misfit` | Float64 | scalar | Best weighted misfit |
| `iteration` | Int32 | scalar | Iteration number |
| `converged` | Int32 | scalar | Convergence flag (0/1) |
| `convergence_reason` | String | scalar | Stop reason (present when converged=1) |
| `freq_accumulated` | Float64 | `[N_frequencies, 3]` | Best SDR per frequency band |
| `freq_misfit_curve` | Float64 | `[N_frequencies, N_freq_test_mechs]` | Misfit vs frequency vs mechanism |
| `depth_misfit_accumulated` | Float64 | `[N_depths]` | Best misfit per depth |

Total trials: `max(nstrike,1) × max(ndip,1) × max(nrake,1) × max(len(depth_indices),1) × max(len(freq_indices),1)`.

### `/trials`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike` | Float64 | `[N_trials]` | Strike angles (°) |
| `dip` | Float64 | `[N_trials]` | Dip angles (°) |
| `rake` | Float64 | `[N_trials]` | Rake angles (°) |
| `depth` | Float64 | `[N_trials]` | Depth (km) |
| `freq_idx` | Int32 | `[N_trials]` | Frequency band index (into `database.h5`) |
| `depth_idx` | Int32 | `[N_trials]` | GF depth index (into `database.h5`) |

### `/misfits`

Raw per-module misfits (no weighting, no aggregation). Each module has a shape natural to its computation:

| Dataset | Type | Shape | Level |
|---------|------|-------|-------|
| `xcorr` | Float64 | `[N_phases × N_trials]` | phase |
| `polarity` | Float64 | `[N_channels × N_trials]` | channel P-polarity |
| `psr` | Float64 | `[N_channels × N_trials]` | channel P/S ratio |

Future: `absshift`, `relshift`, `cap` under `/misfits/`.

---

## 3. `output.h5` — Final Results

### `/solution`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike` | Float64 | scalar | Best-fit strike (°) |
| `dip` | Float64 | scalar | Best-fit dip (°) |
| `rake` | Float64 | scalar | Best-fit rake (°) |
| `depth` | Float64 | scalar | Best-fit depth (km) |
| `moment_tensor` | Float64 | `[6]` | [Mxx,Myy,Mzz,Mxy,Mxz,Myz] |
| `misfit` | Float64 | scalar | Final weighted misfit |

### `/uncertainty`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `strike_std` | Float64 | scalar | Strike uncertainty |
| `dip_std` | Float64 | scalar | Dip uncertainty |
| `rake_std` | Float64 | scalar | Rake uncertainty |
| `depth_range` | Float64 | `[2]` | Depth bounds [min, max] |
| `freq_test_misfit_curve` | Float64 | `[N_frequencies, N_freq_test_mechs]` | Misfit vs frequency |

### `/per_phase`

Phase-level misfit breakdown for the best trial.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `phase_id` | String | `[N_phases]` | Phase identifiers (`NET.STA.COMP.TYPE`) |
| `channel_id` | String | `[N_phases]` | Channel identifier (`NET.STA.COMP`) |
| `station_id` | String | `[N_phases]` | Station identifier (`NET.STA`) |
| `phase_type` | String | `[N_phases]` | `"P"` or `"S"` |
| `misfit_per_module` | Float64 | `[N_modules × N_phases]` | Final misfit per module per phase |
| `selected` | Int32 | `[N_phases]` | Phase selected in final solution |
| `cross_correlation` | Float64 | `[N_phases]` | Best XCorr per phase |

### `/per_station_summary`

Station-level summary across all of a station's channels.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `station_id` | String | `[N_stations]` | Station identifiers |
| `n_channels` | Int32 | `[N_stations]` | Number of channels per station |
| `n_phases` | Int32 | `[N_stations]` | Number of phases per station |
| `mean_cross_correlation` | Float64 | `[N_stations]` | Mean XCorr across station phases |
| `polarity_match` | Int32 | `[N_stations]` | Number of polarity-matching channels |
| `misfit_total` | Float64 | `[N_stations]` | Aggregate misfit per station |

### `/waveforms` (optional)

Present only when waveform synthesis is enabled (e.g., `--waveforms-output` flag). Synthesized from Greens in `database.h5`.

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `{phase_id}` | Float64 | `[N_samples]` | Synthetic seismogram (GF × best_MT) |

### `/summary`

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `total_iterations` | Int32 | scalar | Total iterations |
| `total_trials` | Int32 | scalar | Total trials evaluated |
| `convergence_reason` | String | scalar | Why pipeline stopped |

---

## 4. Signal Conventions

### Convergence Signal

`assess.jl` signals convergence:

- **Continue**: creates `status_{N+1}.h5` with `/strategy/converged=0` and refined grid parameters.
- **Break**: writes `/strategy/converged=1` and `convergence_reason="user"` to the **current** `status_{N}.h5` in-place (no new file is created).

Driver detects convergence by reading `/strategy/converged` from the latest status file.

### Pipeline Stage Detection (Driver)

| File State | Action |
|-----------|--------|
| No `database.h5` | Run `input.jl` (once, with config) |
| `status_{N}.h5` exists, no `/trials` | Run `preprocess.jl` (generate trials from strategy) |
| `status_{N}.h5` exists, has `/trials`, no `/misfits` | Run `forward.cpp` |
| `status_{N}.h5` exists, has `/misfits` | Run `assess.jl` |
| `status_{N}.h5` exists, `/strategy/converged == 1` | Run `output.jl` |

### Config Bootstrap

`config.jl` is a bootstrap-only input read by `input.jl` on the first run. All configuration is written to `database.h5`. Subsequent stages read config from HDF5 only.
