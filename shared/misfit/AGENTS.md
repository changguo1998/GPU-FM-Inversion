# AGENTS.md — Misfit module (`shared/misfit/`)

## Role

Per-module preprocessing for misfit computation. Each module is a template
included inside a dynamically-created Config inner module at pipeline init.
Provides single-trace `preprocess()` and batch `process()` functions.

Active modules: XCorr, Polarity. PSR/AbsShift/RelShift deferred. CAP cancelled.

## Files

| File | Module | Role |
|---------------|----------|----------------------------------------------|
| `Xcorr.jl` | XCorr | Cross-correlation misfit — bandpass + trim |
| `Polarity.jl` | Polarity | Polarity misfit — trim GF to polarity window |

## `process()` return format

Both modules return a `Dict` mirroring the HDF5 schema for direct writing
to `database.h5`. The caller (`input.jl` or a generic Dict→HDF5 writer)
iterates the Dict and writes each key path to the corresponding HDF5 group/dataset.

### XCorr

```
Dict(
    "channel_id"  => String[N_entries],
    "station_idx" => Int32[N_entries],
    "obs" => Dict(
        freq_idx => Dict(
            "obs"       => Float64[N_entries, N_samples],   # filtered + trimmed obs
            "obs_norm2" => Float64[N_entries],               # ‖obs‖² per entry
        ),
    ),
    "gf" => Dict(
        depth => Dict(
            freq_idx => Dict(
                "gf"     => Float64[N_entries, 6, N_samples],  # filtered + trimmed GF
                "synamp" => Float64[N_entries, 6, 6],          # GFᵀ·GF after trim
            ),
        ),
    ),
)
```

`N_samples` = `nt_xc` — the minimum trimmed length across all entries at this
frequency band. All traces are trimmed to this common length so they fit in a
contiguous matrix. `synamp` is recomputed from the nt_xc-trimmed GF (not the
per-entry pre-trimmed GF from `preprocess()`).

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

### Empty entry

When no valid entries exist (missing GF data or no matching phase type), both
modules return the same Dict shape with zero-length arrays:

```
Dict(
    "channel_id"  => String[],
    "station_idx" => Int32[],
    "obs" => Dict(freq_idx => Dict("obs" => zeros(0, 0), "obs_norm2" => Float64[])),
    "gf"  => Dict(d => Dict(freq_idx => Dict("gf" => zeros(0, 6, 0), "synamp" => zeros(0, 6, 6))) for d in depths),
)
```

## Pre-allocation strategy

Both modules use a two-pass approach:

1. **Pass 1** — iterate all phases, check GF availability, collect intermediate
   data in `Vector{Vector}` / `Vector{Matrix}` lists. Track `nt_xc` (XCorr) or
   `n_pol_common` (Polarity) as the minimum length across entries.
1. **Pre-allocate** — allocate `Matrix{Float64}(N_entries, N_samples)` for obs,
   `Array{Float64, 3}(N_entries, 6, N_samples)` for GF per depth, etc.
1. **Pass 2** — copy from intermediate lists into pre-allocated arrays, trimming
   to the common length. Recompute `synamp` from trimmed GF (XCorr only).

This eliminates the O(N²) repeated-array-concatenation that existed in the
original Polarity implementation, while keeping the code readable with
intermediate storage that is garbage-collected after Pass 2.

## Data conventions

- All angles in degrees, all arrays Float64 unless noted.
- `obs_polarity` Int8: `-128` = missing/NaN, `-1/0/1` = observed polarity.
- Phase key format: `{network}.{station}.{channel}.{phase_type}`.
- Station index is 1-based (Julia convention), stored as Int32.
