#ifndef FM_DEVICE_H
#define FM_DEVICE_H

// ─────────────────────────────────────────────────────────
// Device.h — OpenMP backend dispatch
//
// Lightweight OpenMP launcher retained for CPU-only kernels.
// CUDA launchers are defined explicitly in .cu translation units.
//
// Data is stored as flat arrays (double*) with explicit strides.
// No multi-dimensional View abstraction — manual index computation
// gives complete control over memory layout.
//
// Usage: Device<Backend::OpenMP>::parallel_for(n, [=] (int i) { ... });
// ─────────────────────────────────────────────────────────

enum class Backend { OpenMP, CUDA };

// ── OpenMP backend ─────────────────────────────────────────────────────────
template <Backend B> struct Device;

template <> struct Device<Backend::OpenMP> {
    template <typename F> static void parallel_for(int n, F &&f) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            f(i);
        }
    }
};

#endif // FM_DEVICE_H
