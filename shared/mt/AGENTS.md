# AGENTS.md — MT module (`shared/mt/src/MT.jl`)

## Role

SDR (strike/dip/rake) ↔ moment tensor conversion in NED coordinate system. Used by `output.jl` for final solution. Separately reimplemented in C++ `forward/src/mt_utils.*` for GPU-side conversion.

Used by: `output.jl`.

## Exports

| Function | Signature | Returns |
|-------------------|--------------------------|--------------------------------------------------------|
| `sdr_to_mt` | `(strike°, dip°, rake°)` | `Vector{Float64}[6]`: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]` |
| `sdr_to_mt_batch` | `(strikes, dips, rakes)` | `Matrix{Float64}[6 × N]`, each column = one MT |

## Formula

Moment tensor components in NED (North-East-Down) for double-couple source:

```
Mxx = -(sin(2s) sin(d) cos(r) + sin²(s) sin(2d) sin(r))
Myy =  sin(2s) sin(d) cos(r) - cos²(s) sin(2d) sin(r)
Mzz =  sin(2d) sin(r)
Mxy =  cos(2s) sin(d) cos(r) + 0.5 sin(2s) sin(2d) sin(r)
Mxz = -(cos(s) cos(d) cos(r) + sin(s) cos(2d) sin(r))
Myz = -(sin(s) cos(d) cos(r) - cos(s) cos(2d) sin(r))
```

Where: s = strike, d = dip, r = rake (all in radians internally, degrees in API).

Matches legacy `JuliaSourceMechanism.jl` `dc2ts()` implementation.

## Coding conventions

- Input angles in degrees; internally converts to radians via `deg2rad`.
- Pure computation — no I/O, no side effects, no HDF5 dependency.
- C++ `sdr_to_mt()` in `forward/src/mt_utils.*` uses identical formulas (radians input — caller converts). Verified by `tests/test_cross_lang.jl`.
