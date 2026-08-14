# Stage: `forward` (C++) — Raw Intermediate Computation

## Role

Runs once per iteration (preprocess → **forward** → assess). Reads the trial
set from `status_N.h5` and the preprocessed reductions from `database.h5`,
runs the misfit kernels per (freq, depth) combo, and writes RAW INTERMEDIATE
PRODUCTS to `status_N.h5:/intermediates/`. Final misfit values (extract /
compose) are produced by `assess.jl` — this stage never writes `/misfits`.

## Usage

```bash
forward/build/forward <database.h5> <status_N.h5>
```

No other arguments. Exit code 0 on success.

## Inputs

| File | Access | Used for |
|---------------|------------|---------------------------------------------------------------------------|
| `database.h5` | read-only | `/paraspace` axis values, `/config/{Module}`, `/XcorrS/obs|gf`,`/station` |
| `status_N.h5` | read-write | `/trials` (read), `/intermediates/*` (write) |

### Lag half-width is derived from config (not hardcoded)

`maxlag = round(max_lag_periods / band_high_freq / dt)`, with

- `max_lag_periods` from `/config/XcorrS/max_lag_periods`
- `band_high_freq` = `/paraspace/frequency[band_high - 1]`
- `dt` from `/station/dt`

and then clamped per-combo to `(window_len − 1) ÷ 2` inside `DataCache`
(`entry.xcorr.maxlag`). This mirrors `Misfit.Xcorr.preprocess` exactly — there
is no silent truncation relative to the stored 301-lag reductions (2026-08-14).

## XCorr kernel math

For phase `p` and trial `t` (MT `m = sdr_to_mt(strike, dip, rake)`):

```
dot_lag[c]   = Σ_t obs[t+lag]·gf[t,c]                 (obs shifted +lag, zero-padded in window)
synamp_lag   = Σ_t gf[t−lag,a]·gf[t−lag,b]            (GF shifted −lag, same-shift normalization)
cc_norm[lag] = (mᵀ·dot_lag) / sqrt(obs_n2 · mᵀ·synamp_lag·m)
cc_max       = max_lag |cc_norm[lag]|                 (Float64)
best_lag     = argmax_lag |cc_norm[lag]| − maxlag     (Int32, samples, relative shift)
```

The per-lag `synamp` uses the **same shift direction as the dot products**
(GF shifted by −lag), so the normalization obeys Cauchy–Schwarz and `cc ≤ 1`.
Both quantities are recomputed by the C++ from the windowed obs/GF (`/XcorrS/obs`
and `/XcorrS/gf/{depth}/{freq}/gf`) per (freq, depth) combo — the stored
`dot_obs_gf_lag` reduction covers only depth index 1, so per-depth evaluation
cannot consume it.

Verified (2026-08-14): an independent Julia reference recomputing the same
math from the stored windows matches C++ `cc_max`/`best_lag` to ≤1e-9 on every
checked (trial, phase) pair (`tests/stages/forward_test.jl`).

## Outputs

### `status_N.h5:/intermediates/{Operator}{Phase}[_{channel}]/`

Grouped by canonical key (e.g. `XcorrS`). Deduplicated across module instances
sharing one key. **Idempotent**: the whole `/intermediates` group is deleted and
recreated on every run, so re-running forward over the same status file is safe.

| Dataset | Type | Shape (HDF5) | Description |
|------------|---------|-------------------------|--------------------------------------------------|
| `cc_max` | Float64 | `[N_phases × N_trials]` | max normalized CC per (phase, trial) |
| `best_lag` | Int32 | `[N_phases × N_trials]` | best-lag offset in samples (∈ [−maxlag, maxlag]) |

Storage is C-order `[N_phases × N_trials]`; `HDF5.jl` reads it as
`(N_trials, N_phases)` (reversed dims) — account for this in consumers.

## Known behavior notes

- `H5Lexists` probes of optional groups (`/XcorrP/...`) are silenced with
  `H5E_BEGIN_TRY`; forward produces no `HDF5-DIAG` noise on stderr.
- `station_idx` is normalized from 1-based (HDF5) to 0-based immediately after
  reading; the result is consumed only by the (deferred) Polarity path.
- Polarity/PSR kernels remain compiled but are **deferred** — no instances are
  registered in sample configs (XCorr-only mode).

## What It Does NOT Do

- Does NOT write `/misfits` (assess.jl).
- Does NOT resolve trials to physical values beyond SDR→MT conversion
  (angles are resolved from `/paraspace` before the kernel).
- Does NOT apply weights or make convergence decisions.
