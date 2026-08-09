# Module: Signal (Waveform Preprocessing)

**Location**: `shared/signal/` (Julia package `Signal`)

> **当前状态**: `input.jl` 使用 `Signal.preprocess_waveform!()`（Layer 0 共享预处理：demean/detrend/taper/bandpass per band），算子专属窗与 reductions 由 `shared/misfit/` 各算子计算（见 `shared/misfit/AGENTS.md`）。

## Purpose

Filter, trim, and preprocess observed waveforms and Green's functions for each frequency band and misfit module.

## Used By

- `input.jl` (first run only — writes to `database.h5`)

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
|----------|--------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------------|
| XCorr | Layer 0 bandpass + fixed obs window; per-lag reductions | `obs`, `obs_norm2`, `synamp_lag`, `dot_obs_gf_lag` | active |
| Polarity | Trim GF to polarity window `[0, t_source]` | `gf_pol`, `obs_pol` | **deferred** |
| PSR | Compute/store P/S amplitude-ratio observation and reductions | `amp_P`, `amp_S`, `obs_psr` | **deferred** (operator implemented; XCorr-only mode) |
| AbsShift | Spatial component decomposition | `obs[3×N]`, `gf[3×N×6]` | **deferred** |
| RelShift | Spatial component concatenation | `obs[3×N]`, `gf[3×N×6]` | **deferred** |
| CAP | Cut-and-paste waveform fitting | `obs[3×N]`, `gf[3×N×6]` | **cancelled** |

Persistent preprocessing belongs to `input.jl` and `database.h5`; non-persistent reductions (XCorr CC, Polarity pol_vec aggregation) belong to `forward.cpp`'s DataCache and run on the host CPU.

## Testing Strategy

- Filter round-trip: filter known signal, verify frequency content
- Trim accuracy: verify correct sample indices for given window
- Synamp identity: `m'·synamp·m = ‖GF·m‖²` for random m vectors
- PSR: verify envelope computation against known P/S ratios
