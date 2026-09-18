# Module: Misfit Kernels (OpenMP CPU / CUDA)

## Description

Kernels live in `forward/src/kernels/`, all functions in namespace `fm`.
XCorr P+S is active on OpenMP and CUDA. Polarity/PSR templates remain
**deferred** and CUDA rejects configurations containing active non-XCorr operators.

## Used By

- `main.cpp` — launched per `(freq_idx, depth_idx, duration_idx)` combo

## XCorr shared formula and launchers

`xcorr_work_item(...)` is the only XCorr mathematical implementation and is
compiled for host/device. Launchers only map work-items:

```cpp
namespace fm {
void xcorr_work_item(..., int n_phases, int n_trials, int cc_pp, int maxlag, int index);
void launch_xcorr_openmp(...);
void launch_xcorr_cuda(...); // implemented only in xcorr_kernel.cu
} // namespace fm
```

CUDA uses 128 threads/block and one thread per `(phase, trial)`. Both builds use
FP contraction disabled (`-ffp-contract=off`, `--fmad=false`); no fast-math.

## XCorr Kernel

**Kernel outputs (intermediate products — NOT final misfit):**

```
cc_syn[k]   = Σᵢ m[i] · CC[phase][k][i]        // weighted CC sum
syn_norm²   = mᵀ · synamp · m                   // 6×6 quadratic form
cc_norm[k]  = cc_syn[k] / √(obs_norm² · syn_norm²)
best_lag    = argmaxₖ(cc_norm[k]) − maxlag     // → best_lag_out (Int32, samples)
cc_max      = maxₖ(cc_norm[k])                  // signed → cc_max_out
```

每个 work item 在 lag 循环前预计算 21 个唯一的上三角系数
`(i == j ? 1 : 2) · m[i] · m[j]`；每个 lag 只读取对称 Gram 矩阵的上三角。
系数保持为线程局部数据，不新增 HDF5 数据集或 CUDA 全局显存缓冲区。

The final misfit is derived in Julia assess: `misfit = 1.0 − cc_max`; AbsShift
is derived from `best_lag · dt`. The kernel does **not** output a `misfit` array.

**Verification invariant:** `mᵀ·synamp·m = ‖GF·m‖²` (Gram matrix identity). Tests must verify this for random `m` vectors.

**Inputs/outputs:**

```cpp
mt           // [N_trials × 6] ROW-major: mt[trial * 6 + comp]  (2026-08-10 fixed: was mistakenly column-major)
cc_data      // [N_phases · cc_pp × 6] column-major
synamp_data  // [N_phases × 36] column-major: synamp_data[phase + (i*6+j) * N_phases]
obs_norm2    // [N_phases]
cc_max_out   // [N_phases × N_trials] column-major: cc_max_out[phase + trial * N_phases]
best_lag_out // [N_phases × N_trials] Int32 column-major (relative shift, samples)
```

## Polarity Kernel (deferred — not launched in current baseline)

**Outputs (intermediate products):**

```
dot_value = Σᵢ pol_vec[station][i] · mt[trial][i]     // raw dot product → dot_value_out
syn_sign  = sign(dot_value)                            // → syn_sign_out (Int8)
```

Misfit is derived in Julia assess: `misfit = (syn_sign == obs_pol) ? 0.0 : 1.0`.
Missing polarity (obs_pol is NaN, or obs_pol == 0.0 with zero pol_vec) returns
NaN to signal "not applicable".

**Signature (matches polarity_kernel.h):**

```cpp
namespace fm {
template <Backend B>
void launch_polarity_kernel(const double *mt,        // N_trials × 6, column-major
                            const double *pol_vec,   // N_stations × 6, column-major
                            int8_t *syn_sign_out,    // N_stations × N_trials
                            double *dot_value_out,   // N_stations × N_trials
                            int N_stations, int N_trials);
}
```

## PSR Kernel (deferred — never launched from main.cpp)

Header-only template exists in `psr_kernel.h`; `launch_psr_kernel` is defined
but **never called** in the current pipeline (PSR input reductions are not
precomputed in `database.h5`). Signature retained for reference:

```cpp
namespace fm {
template <Backend B>
void launch_psr_kernel(
    const double *mt,      // N_trials × 6, column-major
    const double *amp_P,   // N_stations × 6 × 6, column-major
    const double *amp_S,   // N_stations × 6 × 6, column-major
    const double *obs_psr, // N_stations
    double *misfit,        // N_stations × N_trials, column-major
    int N_stations, int N_trials);
}
```

Missing stations (obs_psr is NaN, or amplitude near zero) returns NaN.

**Misfit formula (Julia assess, when restored):**

```
syn_amp_P = √(mᵀ · amp_P · m)          // synthetic P amplitude
syn_amp_S = √(mᵀ · amp_S · m)          // synthetic S amplitude
misfit = (log₁₀(syn_amp_P / syn_amp_S) - obs_psr)²
```

## Launch Strategy (per combo in main.cpp)

XCorr reductions are uploaded once per combo. CPU evaluates the combo directly;
CUDA packs original (possibly shuffled) trial order into reusable batch buffers,
then scatters returned columns to the full host output:

```cpp
fm::launch_xcorr_openmp(...);
cuda_executor.evaluate(...); // handles one or more batches
```

The CUDA path uses one default stream and synchronizes every batch before D2H.
No generic backend fallback exists: CUDA runtime errors fail the stage and do
not retry on CPU.

## Testing Strategy

- CPU and CUDA P+S outputs match independent Julia reference (`cc_max` ≤1e-9,
  `best_lag` exact)
- Verify linear decomposition identity: `‖GF·m‖² = mᵀ·synamp·m` (Gram matrix identity)
- Boundary: exact/near tie, signed all-negative correlation, zero norm,
  degenerate synthetic norm, maxlag clamp, NaN/Inf preflight rejection
- CUDA: auto/1/prime/residual batches, shuffled multi-combo trials, injected
  first/middle-batch failure, idempotence and compute-sanitizer memcheck
- Polarity: all sign combos, edge cases (NaN, zero, ambiguous)
- PSR: hand-calculated, non-diagonal `amp` matrices, degenerate zero-amplitude
- Combined back-to-back launch (Polarity + PSR in one test) — only when restored
