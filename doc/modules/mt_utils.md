# Module: MT (SDR ↔ MT Conversion)

**Location**: `shared/mt/` (Julia package `MT`)

> **当前状态**: Julia 与 C++ 实现均在使用。forward 在 host 将每个 trial 的 SDR 转为 MT，再把按 combo 打包的 MT 交给 OpenMP 或 CUDA XCorr launcher。

## Description

Double-couple SDR (strike, dip, rake) to 6-component moment tensor conversion. Must produce identical results in both Julia and C++ to 6 decimal places.

## Used By

| Stage | Language |
|------------------------|---------------------------------|
| `forward/src/main.cpp` | C++（host SDR → MT，CPU/GPU 共用） |
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

**Julia (`MT` module):**

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

```

Angles are in **radians**. No batch interface in C++ — the Julia side generates arrays and the C++ side iterates per trial.

## Verification

- Both implementations verified against each other on multiple random SDR inputs via cross-language CSV roundtrip (`test_mt_to_csv.cpp` + `test_cross_lang.cpp` `--mode mt-csv`).
- Maximum absolute difference across all 6 components: < 1e-12 (double precision).
- Unit test: compare Julia and C++ output on canonical cases (strike=0/dip=90/rake=0 for pure double-couple).

## GPU/CPU Notes

- MT 转换只在 host 执行一次，不在 CUDA kernel 内重复计算。
- XCorr MT 输入为 **row-major** `mt[trial * 6 + comp]`；combo 与 batch 打包都保持此布局。
- CPU 与 CUDA 消费同一 MT 数组和同一 `xcorr_work_item` 数学实现。
