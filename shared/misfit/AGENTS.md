# AGENTS.md — Misfit module (`shared/misfit/`)

## Role

Per-module preprocessing for misfit computation. Each module is a template
included inside a dynamically-created Config inner module at pipeline init.
Provides single-trace `preprocess()` and batch `process()` functions.

Operators: Xcorr, Polarity, Psr. AbsShift = Xcorr BEST_LAG output; RelShift = Aggregate.StdDev composer (registered in sample configs). PSR operator implemented but no instance registered in sample configs. CAP cancelled.

## Files

| File | Module | Role |
|-----------------|--------|------------------------------------------------------------------------|
| `src/Misfit.jl` | Misfit | Package entry, wraps Xcorr/Polarity/Psr as sub-modules |
| `src/Xcorr.jl` | XCorr | Cross-correlation misfit - bandpass + trim + outputs() |
| `src/Psr.jl` | Psr | P/S amplitude-ratio misfit - log10(rms_P/rms_S) reductions + outputs() |

## `process()` return format

All modules return a `Dict` mirroring the HDF5 schema for direct writing
to `database.h5`. The caller (`input.jl` or a generic Dict→HDF5 writer)
iterates the Dict and writes each key path to the corresponding HDF5 group/dataset.

### XCorr

```
Dict(
    "channel_id"      => String[N_entries],
    "station_idx"     => Int32[N_entries],
    "obs" => Dict(
        freq_idx => Dict(
            "obs"       => Float64[N_entries, nt_win],   # filtered + trimmed obs window
            "obs_norm2" => Float64[N_entries],           # ‖obs‖² per entry
        ),
    ),
    "synamp_lag"      => Dict(depth => Dict(freq_idx => Float64[N_entries, 6, 6, L])),
    "dot_obs_gf_lag"  => Dict(freq_idx => Float64[N_entries, 6, L]),
    "gf" => Dict(                                           # (debug) trimmed GF at lag=0
        depth => Dict(freq_idx => Dict("gf" => Float64[N_entries, 6, nt_win])),
    ),
)
```

`nt_win` = `pre_n + post_n + 1` — the fixed obs window length (`pre_periods`/`post_periods`
wavelengths scaled by `1/band_high`). `L = 2 * max_lag_n + 1` is the per-lag count;
`synamp_lag[l] = gf[win-l]ᵀ·gf[win-l]` (6×6) and `dot_obs_gf_lag[l] = obsᵀ·gf[win-l]`
(6-vector) are per-lag reductions consumed by the forward kernel at runtime. The
`gf` trimmed-window key is debug-only (leftover from development, to be removed).

### Polarity

```
Dict(
    "channel_id"  => String[N_entries],
    "station_idx" => Int32[N_entries],
    "obs" => Dict(
        1 => Dict(
            "obs"       => Float64[N_entries],   # polarity values (1.0, -1.0, NaN)
        ),
    ),
    "gf" => Dict(
        depth => Dict(
            1 => Dict(
                "gf"     => Float64[N_entries, 6, N_pol],   # GF in polarity window
            ),
        ),
    ),
)
```

Polarity uses band index 1 (no frequency filtering). `N_pol` is the minimum
polarity window length across all entries; longer windows are trimmed.

### PSR

```
Dict(
    "channel_id"  => String[N_entries],
    "station_idx" => Int32[N_entries],
    "obs"   => Dict(freq_idx => Dict("obs_psr" => Float64[N_entries])),
    "amp_P" => Dict(depth => Dict(freq_idx => Float64[N_entries, 6, 6])),
    "amp_S" => Dict(depth => Dict(freq_idx => Float64[N_entries, 6, 6])),
)
```

`obs_psr` = `log10(rms_P / rms_S)` amplitude ratio per entry; `amp_P`/`amp_S` are the
GFᵀ·GF Gram matrices within the P/S windows per depth. PSR is freq-dependent
(`is_freq_dependent() = true`), one entry per station with both P and S picks.

### Empty entry

When no valid entries exist (missing GF data or no matching phase type), each
module returns its own Dict shape with zero-length arrays. Example (XCorr):

```
Dict(
    "channel_id"  => String[],
    "station_idx" => Int32[],
    "obs" => Dict(freq_idx => Dict("obs" => zeros(0, 0), "obs_norm2" => Float64[])),
    "synamp_lag"      => Dict(d => Dict(freq_idx => zeros(Float64, 0, 6, 6, 0)) for d in depths),
    "dot_obs_gf_lag"  => Dict(freq_idx => zeros(Float64, 0, 6, 0)),
    "gf"  => Dict(d => Dict(freq_idx => Dict("gf" => zeros(Float64, 0, 6, 0))) for d in depths),
)
```

## Pre-allocation strategy

All modules use a two-pass approach:

1. **Pass 1** — iterate all entries, check GF availability, collect intermediate
   data in `Vector{Vector}` / `Vector{Matrix}` lists. Track `nt_win` (XCorr) or
   `N_pol` (Polarity) per entry.
1. **Pre-allocate** — allocate `Matrix{Float64}(N_entries, N_samples)` for obs,
   `Array{Float64, 3}(N_entries, 6, N_samples)` for GF per depth, etc.
1. **Pass 2** — copy from intermediate lists into pre-allocated arrays.

This eliminates the O(N²) repeated-array-concatenation that existed in the
original Polarity implementation, while keeping the code readable with
intermediate storage that is garbage-collected after Pass 2.

## Data conventions

- All angles in degrees, all arrays Float64 unless noted.
- `obs_polarity` Int8: `-128` = missing/NaN, `-1/0/1` = observed polarity.
- Phase key format: `{network}.{station}.{channel}.{phase_type}`.
- Station index is 1-based (Julia convention), stored as Int32.
