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
// forward/src/mt_utils.h
struct MomentTensor {
    double Mxx, Myy, Mzz, Mxy, Mxz, Myz;
};

// Host function (declared in header, defined in .cpp)
MomentTensor sdr_to_mt(double strike_rad, double dip_rad, double rake_rad);

// GPU-compatible device function (also callable from host)
MT_HOST_DEVICE MomentTensor sdr_to_mt_device(double strike_rad, double dip_rad, double rake_rad);
```

Angles are in **radians**. No batch interface in C++ — the Julia side generates arrays and the C++ side iterates per trial.

## Verification

- Both implementations verified against each other on multiple random SDR inputs via cross-language CSV roundtrip (`test_mt_to_csv.cpp` + `test_cross_lang.cpp` `--mode mt-csv`).
- Maximum absolute difference across all 6 components: < 1e-12 (double precision).
- Unit test: compare Julia and C++ output on canonical cases (strike=0/dip=90/rake=0 for pure double-couple).

## GPU/CPU Notes

- `sdr_to_mt_device` is marked `MT_HOST_DEVICE` (expands to `__host__ __device__` under `__CUDACC__`, empty otherwise) for use in both CUDA kernel launches and OpenMP parallel loops (single source, dual-compile).
- Per-trial SDR→MT conversion may happen during kernel launch or in a separate pre-conversion pass.
- When pre-converting all trials to `mt[N_trials × 6]`, the conversion runs on device via `Device<Backend>::parallel_for` — same dispatch pattern as the misfit kernels.
- Flat arrays with explicit strides replace `Kokkos::View`. All data is column-major `double*` with manual index computation.