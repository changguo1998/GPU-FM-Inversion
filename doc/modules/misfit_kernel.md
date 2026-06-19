# Module: Misfit Kernels (GPU/CPU)

## Description

Parallel kernels for computing per-module misfits. Each kernel operates on a flat grid over `(phase × trial)` work items (XCorr) or `(station × trial)` work items (Polarity/PSR). Kernels are templated on `Backend` and dispatched via `Device<B>::parallel_for` — single source compiles to both OpenMP (`#pragma omp parallel for`) and CUDA (`__global__` kernel). No external GPU framework dependency.

All kernel functions live in the `fm` namespace and are header-only (`forward/src/kernels/`).

## Used By

- `forward.cpp` — launched after data precomputation

## Backend Dispatch Pattern

All kernels follow the same structure — template function wrapping a `Device<B>::parallel_for` call:

```cpp
namespace fm {
template <Backend B>
void launch_xcorr_misfit(
    const double* mt,           // N_trials × 6, column-major
    const double* cc_data,      // N_phases × (2·maxlag+1) × 6
    const double* synamp_data,  // N_phases × 36
    const double* obs_norm2,    // N_phases
    double* misfit,             // N_phases × N_trials
    int N_phases, int N_trials, int cc_pp);
}  // namespace fm
```

## XCorr Kernel

**Misfit formula:**
```
cc_syn[k] = Σᵢ m[i] · CC[phase][k][i]        // weighted CC sum
syn_norm² = mᵀ · synamp · m                   // 6×6 quadratic form
cc_norm[k] = cc_syn[k] / √(obs_norm² · syn_norm²)
misfit = 1.0 - maxₖ(|cc_norm[k]|)
```

**Verification invariant:** `mᵀ·synamp·m = ‖GF·m‖²` (Gram matrix identity). Tests must verify this for random `m` vectors.

**Inputs/outputs:**
```cpp
mt           // [N_trials × 6] column-major: mt[trial + comp * N_trials]
cc_data      // [N_phases · cc_pp × 6] column-major
synamp_data  // [N_phases × 36] column-major: synamp_data[phase + (i*6+j) * N_phases]
obs_norm2    // [N_phases]
misfit       // [N_phases × N_trials] column-major: misfit[phase + trial * N_phases]
```

## Polarity Kernel

**Misfit formula:**
```
syn_pol = sign(Σᵢ pol_vec[station][i] · mt[trial][i])
misfit  = (syn_pol == obs_pol) ? 0.0 : 1.0
```

Missing polarity (obs_pol is NaN, or obs_pol == 0.0 with zero pol_vec) returns NaN to signal "not applicable".

**Signature:**
```cpp
namespace fm {
template <Backend B>
void launch_polarity_kernel(
    const double* mt,       // N_trials × 6, column-major
    const double* pol_vec,  // N_stations × 6, column-major
    const double* obs_pol,  // N_stations
    double* misfit,         // N_stations × N_trials, column-major
    int N_stations, int N_trials);
}
```

## PSR Kernel

**Misfit formula:**
```
syn_amp_P = √(mᵀ · amp_P · m)          // synthetic P amplitude
syn_amp_S = √(mᵀ · amp_S · m)          // synthetic S amplitude
misfit = (log₁₀(syn_amp_P / syn_amp_S) - obs_psr)²
```

**Signature:**
```cpp
namespace fm {
template <Backend B>
void launch_psr_kernel(
    const double* mt,       // N_trials × 6, column-major
    const double* amp_P,    // N_stations × 6 × 6, column-major
    const double* amp_S,    // N_stations × 6 × 6, column-major
    const double* obs_psr,  // N_stations
    double* misfit,         // N_stations × N_trials, column-major
    int N_stations, int N_trials);
}
```

Missing stations (obs_psr is NaN, or amplitude near zero) returns NaN.

## Launch Strategy

All three kernels launched back-to-back from `main.cpp`:

```cpp
fm::launch_xcorr_misfit<Backend::OpenMP>(mt, cc, synamp, obs_n2, mis_xcorr, Nph, Ntr, cc_pp);
fm::launch_polarity_kernel<Backend::OpenMP>(mt, pol_vec, obs_pol, mis_pol, Nst, Ntr);
fm::launch_psr_kernel<Backend::OpenMP>(mt, ampP, ampS, obs_psr, mis_psr, Nst, Ntr);
```

With CUDA, substitute `Backend::CUDA`. OpenMP has implicit barriers after each `parallel_for`; CUDA requires explicit `cudaDeviceSynchronize()` between dissimilar kernel types.

## Testing Strategy

- Kernel output matches reference CPU implementation (`ref_misfit` in `test_xcorr.cpp`)
- Verify linear decomposition identity: `‖GF·m‖² = mᵀ·synamp·m` (Gram matrix identity)
- Boundary: zero-norm, maxlag=0, single phase, single trial
- Polarity: all sign combos, edge cases (NaN, zero, ambiguous)
- PSR: hand-calculated, non-diagonal `amp` matrices, degenerate zero-amplitude
- Combined back-to-back launch (Polarity + PSR in one test)
