#ifndef FM_DEVICE_H
#define FM_DEVICE_H

// ─────────────────────────────────────────────────────────
// Device.h — Custom backend dispatch template
//
// Lightweight alternative to Kokkos. Provides a Device<Backend>
// struct with a static parallel_for method. Only two backends
// are needed: OpenMP (host CPU, #pragma omp parallel for) and
// CUDA (__global__ kernel launch).
//
// Data is stored as flat arrays (double*) with explicit strides.
// No multi-dimensional View abstraction — manual index computation
// gives complete control over memory layout.
//
// Usage:
//   Device<Backend::OpenMP>::parallel_for(n, [=] (int i) { ... });
//   Device<Backend::CUDA>::parallel_for(n, [=] __device__ (int i) { ... });
// ─────────────────────────────────────────────────────────

#include <cmath>
#include <cstdint>

enum class Backend {
    OpenMP,
    CUDA
};

// ── Host math functions ────────────────────────────────────────────────────
namespace detail {
    inline double host_isnan(double x) { return std::isnan(x); }
    inline double host_fabs(double x) { return std::fabs(x); }
    inline double host_sqrt(double x) { return std::sqrt(x); }
    inline double host_log10(double x) { return std::log10(x); }
} // namespace detail

// ── OpenMP backend ─────────────────────────────────────────────────────────
template <Backend B>
struct Device {
    // Default (non-CUDA) implementation uses OpenMP
    template <typename F>
    static void parallel_for(int n, F&& f) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            f(i);
        }
    }
};

// ── CUDA backend ───────────────────────────────────────────────────────────
#ifdef __CUDACC__
#include <cuda_runtime.h>

template <typename F>
__global__ void device_parallel_for_kernel(F f, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f(i);
}

template <>
struct Device<Backend::CUDA> {
    template <typename F>
    static void parallel_for(int n, F f) {
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        device_parallel_for_kernel<<<grid_size, block_size>>>(f, n);
    }
};
#endif

// ── Default backend ────────────────────────────────────────────────────────
// With CUDA enabled, CUDA is the default; otherwise OpenMP.
#if defined(__CUDACC__)
    using DefaultDevice = Device<Backend::CUDA>;
#else
    using DefaultDevice = Device<Backend::OpenMP>;
#endif

#endif // FM_DEVICE_H