# Module: MT Utils (SDR ↔ MT Conversion)

## Description

Double-couple SDR (strike, dip, rake) to 6-component moment tensor conversion. Must produce identical results in both Julia and C++ to 6 decimal places.

## Used By

| Stage | Language |
|-------|----------|
| `setup.jl` | Julia (generating trials) |
| `forward.cpp` | C++ (SDR → MT on GPU, per trial) |
| `export.jl` | Julia (recomputing best-fit MT) |

## Algorithm

```
Input: strike ∈ [0,360), dip ∈ [0,90], rake ∈ [-90,90]  (degrees)

Convert to radians, compute:
    s = sin(strike), c = cos(strike)
    d = sin(dip),   d' = cos(dip)
    r = sin(rake),  r' = cos(rake)

Mxx = -(s·d·r' + c·s·d·r + c·d'·r + s·d'·r)
Myy =  (c·d·r' - s·s·d·r - s·d'·r + c·d'·r)
Mzz =  d·r
Mxy = -(c·d·r' - s·s·d·r + c·d'·r - s·d'·r)
Mxz = -(c·d'·r' - c·d·r - s·d'·r' + s·d·r)
Myz =  s·d'·r' - c·d'·r' - c·d·r - s·d·r
```

Output: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]` in NED coordinate system.

## Interface

**Julia (MTUtils package):**
```julia
function sdr_to_mt(strike::Float64, dip::Float64, rake::Float64)::Vector{Float64}
# Returns 6-element vector. Batch version:
function sdr_to_mt_batch(strikes::Vector{Float64}, dips::Vector{Float64}, rakes::Vector{Float64})::Matrix{Float64}
# shape: [6, N_trials]
```

**C++ (header only, shared):**
```cpp
// src/mt_utils.h — includes in both trial_reader.cpp and kernels/
struct MomentTensor {
    float Mxx, Myy, Mzz, Mxy, Mxz, Myz;
};

MomentTensor sdr_to_mt(float strike_rad, float dip_rad, float rake_rad);

// Batch version for GPU — called with RangePolicy
__host__ __device__ MomentTensor sdr_to_mt_device(float strike, float dip, float rake);
```

## Verification

- Both implementations verified against each other on 100 random SDR inputs.
- Maximum absolute difference across all 6 components: < 1e-6.
- Unit test: compare Julia and C++ output on canonical cases (strike=0/dip=90/rake=0 for pure double-couple).

## GPU Notes

- `sdr_to_mt_device` is marked `__host__ __device__` for inline Kokkos kernel use.
- Per-trial conversion happens during kernel launch — no separate pre-conversion step needed.
- Alternatively, pre-convert all trials to `mt[6 × N_trials]` on host/GPU before kernel launch (used by DataCache).