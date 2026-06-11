# Module: Waveform Preprocessing

## Purpose

Filter, trim, and preprocess observed waveforms and Green's functions for each frequency band and misfit module.

## Used By

- `setup.jl` (first run only — writes to `database.h5`)

## Operations

### 1. Bandpass Filtering

- Butterworth filter (order from config, default 4)
- Zero-phase (forward-backward) to preserve alignment
- One filter per frequency band

### 2. Time-Window Trimming

- Window defined as wavelength-multiplier factors in config
- Converted to seconds: `window_seconds = window_factor / band_high`
- Relative to P or S arrival time
- Applied to both observed and Green's function waveforms

### 3. Per-Module Preprocessing

| Module | Operation | Output | Status |
|--------|-----------|--------|--------|
| XCorr | Filter/trim waveform pair; store any persistent reduction chosen by schema | `obs`, `gf`, optional `synamp` | active |
| Polarity | Trim GF to polarity window `[0, t_source]` | `gf_pol`, `obs_pol` | active |
| PSR | Compute/store P/S observation and any persistent covariance chosen by schema | `amp_P`, `amp_S`, `obs_psr` | active |
| AbsShift | Spatial component decomposition | `obs[3×N]`, `gf[3×N×6]` | **deferred** |
| RelShift | Spatial component concatenation | `obs[3×N]`, `gf[3×N×6]` | **deferred** |
| CAP | Cut-and-paste waveform fitting | `obs[3×N]`, `gf[3×N×6]` | **deferred** |

Persistent preprocessing belongs to `setup.jl` and `database.h5`; non-persistent GPU reductions belong to `forward.cpp`'s DataCache.

## Testing Strategy

- Filter round-trip: filter known signal, verify frequency content
- Trim accuracy: verify correct sample indices for given window
- Synamp identity: `m'·synamp·m = ‖GF·m‖²` for random m vectors
- PSR: verify envelope computation against known P/S ratios
