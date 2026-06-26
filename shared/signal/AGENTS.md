# AGENTS.md — Signal module (`shared/signal/src/Signal.jl`)

## Role

Waveform preprocessing: filtering, time-window trimming, per-module preprocessing. Called by `input.jl` for each phase during data ingestion. Pure computation — no HDF5 I/O.

Used by: `input.jl`.

## Dependencies

- `DSP.jl` — digital filter design + `filtfilt` (zero-phase)
- `FFTW.jl` — FFT for Hilbert envelope
- `LinearAlgebra`, `Statistics`

## Exports

### Filtering

| Function | Role |
|-------------------------------------------------------|-------------------------------------------------|
| `bandpass_filter!(x, dt, low_cut, high_cut; order=4)` | Zero-phase Butterworth bandpass filter in-place |

Butterworth `order=4`. Zero-phase via forward-backward `filtfilt`. Clamps high cut to 0.999×Nyquist, low cut to ≥1e-6. No-op if low ≥ high.

### Trimming

| Function | Role |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `trim_time_window!(obs, gf, dt, arrival_sample, window_factor, band_high)` | Trim obs/gf to time window around arrival. Window = `(window_factor / band_high)` seconds. Returns trimmed arrays. |
| `trim_to_polarity_window!(gf, dt, arrival_sample, t_source)` | Trim GF to `[arrival, arrival + t_source]` window. Returns trimmed matrix. |

### Per-module preprocessing

| Function | Input | Output | Used for |
|------------------------|-------------------------------------------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------|
| `preprocess_xcorr!` | obs waveform, GF matrix, dt, arrival_sample, low_cut, high_cut, window_factor | `(obs_proc, gf_proc, synamp[6×6], obs_norm2)` | XCorr module — filtered + trimmed obs/GF + auto-correlation matrix |
| `preprocess_polarity!` | GF matrix, dt, arrival_sample, t_source, obs_polarity(Int8) | `(gf_pol[N_pol×6], obs_pol_float)` | Polarity module — GF in polarity window; -128→NaN |
| `preprocess_psr!` | obs_P, obs_S, GF_P, GF_S, dt, arrival_P/S, pre/post seconds | `(amp_P[6×6], amp_S[6×6], obs_psr)` | PSR module — GF auto-correlation matrices + log10 RMS ratio |

### Utilities

| Function | Role |
|--------------------|------------------------------------------------------|
| `envelope(x)` | Hilbert envelope (analytic signal magnitude) via FFT |
| `rms_amplitude(x)` | Root-mean-square amplitude |

## Preprocessing sequence (per phase, per freq band)

1. `bandpass_filter!` on obs and each GF component
1. `trim_time_window!` for XCorr — centered on arrival, scaled by `window_factor / high_cut`
1. Compute `synamp = gf'·gf` (6×6 Gram matrix) + `obs_norm2 = ‖obs‖²`
1. `trim_to_polarity_window!` for Polarity — GF for P-wave only
1. PSR data (`preprocess_psr!`) not called by current `input.jl` — C++ DataCache handles PSR data when present in database.h5

## Coding conventions

- All functions mutate/filter in-place where practical (noted by `!` suffix).
- `obs_polarity` Int8: `-128` = missing/NaN, `-1/0/1` = observed polarity.
- No HDF5 I/O, no pipeline state — pure signal processing.
