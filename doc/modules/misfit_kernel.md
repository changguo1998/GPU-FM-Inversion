# Module: Misfit Kernels (OpenMP CPU; CUDA-ready)

## Description

Header-only kernels in `forward/src/kernels/`, all functions in the `fm`
namespace, templated on `Backend` (`OpenMP` active; `CUDA` behind `__CUDACC__`).
XCorr is the active operator (XCorrS-only pipeline); Polarity/PSR kernels exist
as templates but are **deferred** — not launched from `main.cpp`.

## Used By

- `main.cpp` — launched per (freq_idx, depth_idx) combo after data precomputation

## Backend Dispatch Pattern

All kernels follow the same structure — a template function wrapping a
`Device<B>::parallel_for` call:

```cpp
namespace fm {
template <Backend B>
inline void launch_xcorr_misfit(const double *mt,            // N_trials × 6, ROW-major
                                const double *cc_data,      // [N_phases·cc_pp × 6] column-major
                                const double *synamp_data,  // [N_phases × 36] column-major
                                const double *obs_norm2,    // [N_phases]
                                double *cc_max_out,         // [N_phases × N_trials] column-major
                                int32_t *best_lag_out,      // [N_phases × N_trials] column-major
                                int N_phases, int N_trials, int cc_pp, int maxlag);
}  // namespace fm
```

## XCorr Kernel

**Kernel outputs (intermediate products — NOT final misfit):**

```
cc_syn[k]   = Σᵢ m[i] · CC[phase][k][i]        // weighted CC sum
syn_norm²   = mᵀ · synamp · m                   // 6×6 quadratic form
cc_norm[k]  = cc_syn[k] / √(obs_norm² · syn_norm²)
cc_max      = maxₖ(|cc_norm[k]|)                // → cc_max_out
best_lag    = argmaxₖ(|cc_norm[k]|) − maxlag    // → best_lag_out (Int32, samples)
```

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

## Polarity Kernel (deferred — not launched in XCorr-only mode)

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
but **never called** in the XCorr-only pipeline (PSR input reductions are not
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

Kernels are launched on each (freq_idx, depth_idx) combo's trial subset.
Currently only XCorr is active:

```cpp
fm::launch_xcorr_misfit<Backend::OpenMP>(
    mt_xcorr_sub.data(), entry->xcorr.cc, synamp_r.data(), entry->xcorr.obs_norm2,
    cc_max_sub.data(), best_lag_sub.data(), N_phases, n_sub, cc_pp, maxlag);
// conditionally, when polarity data is present:
// fm::launch_polarity_kernel<Backend::OpenMP>(mt_pol_sub.data(), pol_vec_s.data(),
//                                             syn_sign_sub.data(), dot_sub.data(),
//                                             N_stations, n_sub);
// launch_psr_kernel is NOT called in the current pipeline (PSR deferred)
```

With CUDA, substitute `Backend::CUDA`. OpenMP has implicit barriers after each
`parallel_for`; CUDA requires explicit `cudaDeviceSynchronize()` between
dissimilar kernel types.

## Testing Strategy

- Kernel output matches reference CPU implementation (per-phase `cc_max`)
- Verify linear decomposition identity: `‖GF·m‖² = mᵀ·synamp·m` (Gram matrix identity)
- Boundary: zero-norm, maxlag=0, single phase, single trial
- Polarity: all sign combos, edge cases (NaN, zero, ambiguous)
- PSR: hand-calculated, non-diagonal `amp` matrices, degenerate zero-amplitude
- Combined back-to-back launch (Polarity + PSR in one test) — only when restored
