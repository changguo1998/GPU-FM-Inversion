# Module: Misfit Kernels (GPU)

## Description

Kokkos parallel kernels for computing per-module misfits. Each kernel operates on a flat `RangePolicy` over `(phase × trial)` work items (XCorr) or `(station × trial)` work items (Polarity/PSR).

## Used By

- `forward.cpp` — launched after data precomputation

## XCorr Kernel

**Misfit formula:**
```
cc_syn[k] = Σᵢ m[i] · CC[phase][k][i]        // weighted CC sum
syn_norm² = mᵀ · synamp · m                   // 6×6 quadratic form
cc_norm[k] = cc_syn[k] / √(obs_norm² · syn_norm²)
misfit = 1.0 - maxₖ(|cc_norm[k]|)
```

**Verification invariant:** `mᵀ·synamp·m = ‖GF·m‖²` (Gram matrix identity). Tests must verify this for random `m` vectors.

**Kokkos inputs/outputs:**
```cpp
mt           // [6 × N_trials] column-major
cc_data      // [N_phases × (2·maxlag+1) × 6]
synamp_data  // [N_phases × 6 × 6]
obs_norm2    // [N_phases]
misfit       // [N_phases × N_trials]
```

## Polarity Kernel

**Misfit formula:**
```
syn_pol = sign(Σᵢ pol_vec[i] · m[i])    // polarity from synthetic GF
misfit = (syn_pol == obs_pol) ? 0.0 : 1.0
```

**Kokkos inputs/outputs:**
```cpp
mt       // [6 × N_trials]
pol_vec  // [N_stations × 6]
obs_pol  // [N_stations]
misfit   // [N_stations × N_trials]
```

## PSR Kernel

**Misfit formula:**
```
syn_amp_P = √(mᵀ · amp_P · m)          // synthetic P amplitude
syn_amp_S = √(mᵀ · amp_S · m)          // synthetic S amplitude
misfit = (log₁₀(syn_amp_P / syn_amp_S) - obs_psr)²
```

**Kokkos inputs/outputs:**
```cpp
mt       // [6 × N_trials]
amp_P    // [N_stations × 6 × 6]
amp_S    // [N_stations × 6 × 6]
obs_psr  // [N_stations]
misfit   // [N_stations × N_trials]
```

## Launch Strategy

All three kernels are launched as labeled Kokkos `parallel_for` work, back-to-back:

```cpp
// After data cache precomputation is complete
Kokkos::parallel_for("xcorr_misfit", policy, KOKKOS_LAMBDA(const int i) { ... });
Kokkos::parallel_for("polarity_misfit", policy, KOKKOS_LAMBDA(const int i) { ... });
Kokkos::parallel_for("psr_misfit", policy, KOKKOS_LAMBDA(const int i) { ... });
Kokkos::fence();

// Copy results back to host
H5Dwrite(misfit_dataset, ...)
```

## Testing Strategy

- Kernel output matches Julia CPU implementation for small test cases
- Verify linear decomposition identity: `‖GF·m‖² = mᵀ·synamp·m`
- Boundary: maxlag=0, single phase, single trial
- Large case: compare GPU vs Julia aggregation on full event
