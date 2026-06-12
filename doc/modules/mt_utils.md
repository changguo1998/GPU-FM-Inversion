# Module: MT Utils (SDR ↔ MT Conversion)

## Description

Double-couple SDR (strike, dip, rake) to 6-component moment tensor conversion. Must produce identical results in both Julia and C++ to 6 decimal places.

## Used By

| Stage | Language |
|-------|----------|
| `input.jl` | Julia (preprocessing; initial strategy) |
| `preprocess.jl` | Julia (generating trials) |
| `forward.cpp` | C++ (SDR → MT on GPU, per trial) |
| `output.jl` | Julia (recomputing best-fit MT) |

## Algorithm

```
Input: strike ∈ [0,360), dip ∈ [0,90], rake ∈ [-90,90]  (degrees)

Convert to radians. Verified reference formulas (from old Julia code mathematics.jl):

    s = strike(rad), d = dip(rad), r = rake(rad)

Mxx = -[sin(2s)·sin(d)·cos(r) + sin²(s)·sin(2d)·sin(r)]
Myy =  sin(2s)·sin(d)·cos(r) - cos²(s)·sin(2d)·sin(r)
Mzz =  sin(2d)·sin(r)
Mxy =  cos(2s)·sin(d)·cos(r) + 0.5·sin(2s)·sin(2d)·sin(r)
Mxz = -[cos(s)·cos(d)·cos(r) + sin(s)·cos(2d)·sin(r)]
Myz = -[sin(s)·cos(d)·cos(r) - cos(s)·cos(2d)·sin(r)]
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