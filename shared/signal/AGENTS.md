# AGENTS.md — Signal module (`shared/signal/src/Signal.jl`)

## Role

Waveform preprocessing: demeaning, detrending, tapering, bandpass filtering, time-window trimming (Layer 0 shared preprocessing consumed by `input.jl`). Per-module reductions live in `shared/misfit/` operators, which call back into these primitives. Pure computation — no HDF5 I/O.

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

### Layer 0 shared preprocessing

| Function | Input | Output | Used for |
|------------------------|-----------------------------------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| `preprocess_waveform!` | waveform, dt, low_cut, high_cut (0/0 = no bandpass) | filtered waveform (in-place option) | XCorr/PSR per-band obs + GF preprocessing (with `do_bandpass=false` → basic clean for Polarity) |

`preprocess_waveform!` runs `demean!` → `detrend!` → `taper!` then
`bandpass_filter!` (unless `do_bandpass=false`). `input.jl` applies it per band
to every observed trace and every GF component (Layer 0); operator-specific
windows/reductions are computed afterwards by the `shared/misfit/` modules.

### Utilities

| Function | Role |
|--------------------|------------------------------------------------------|
| `envelope(x)` | Hilbert envelope (analytic signal magnitude) via FFT |
| `rms_amplitude(x)` | Root-mean-square amplitude |

## Preprocessing sequence (input.jl, per freq band)

1. **Layer 0** — `preprocess_waveform!` (demean/detrend/taper + bandpass) on every obs trace and each GF component of every freq-dependent module's bands; `do_bandpass=false` for the Polarity (non-freq-dependent) basic-clean GF.
1. XCorr/Psr `preprocess()`/`process()` compute operator windows and per-lag / Gram-matrix / amplitude-ratio reductions (see `shared/misfit/AGENTS.md`).
1. `trim_time_window!` / `trim_to_polarity_window!` are called from within the XCorr/Polarity operators.

## Coding conventions

- All functions mutate/filter in-place where practical (noted by `!` suffix).
- `obs_polarity` Int8: `-128` = missing/NaN, `-1/0/1` = observed polarity.
- No HDF5 I/O, no pipeline state — pure signal processing.
