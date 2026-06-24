# Stage: `forward.cpp` — GPU Misfit Computation

## Role

GPU-accelerated core. Stateless misfit computation: reads preprocessed data + trial params, computes raw per-module misfits. No weighting, no aggregation, no strategy knowledge.

## Inputs

| Source | Description |
|--------|-------------|
| `database.h5` | Preprocessed data (read-only): waveform variants, index, Greens |
| `status_{N}.h5` | Trials (`/trials` group): parameters + data slice references |

## Outputs

| Source | Description |
|--------|-------------|
| `status_{N}.h5` | Misfits (`/misfits/` group): one dataset per module, raw (unweighted) |

## Responsibilities

1. **Read inputs**: load trials and preprocessed data from HDF5
2. **Transfer to device**: move all needed data to GPU (or keep on host for OpenMP)
3. **Precompute on host (CPU)**: module-specific reduction via `DataCache` (time-domain CC for XCorr, summed GF for Polarity, amplitude covariances for PSR). All reductions run on the CPU.
4. **Launch kernels**: for each enabled module, launch misfit kernel (trial × phase grid) via backend dispatch
5. **Write results**: copy misfits back to host, write to `status_{N}.h5`

## Execution Model

- Load all data once → precompute on CPU (DataCache) → transfer reduced data to device → launch kernels back-to-back → write
- No data movement between modules (all reduction data fits in GPU memory)
- Linear decomposition: precompute `CC(obs, GF[:,i])` per phase; per-trial: weighted sum of precomputed CCs

## Tool Stack

- C++17 — no external GPU framework dependency
- **Custom backend dispatch**: thin template layer that compiles to OpenMP `#pragma omp parallel for` on CPU, or CUDA kernel launches on GPU. No Kokkos, no heavyweight abstraction — just `Device<Backend>::parallel_for(n, lambda)`. Only OpenMP and CUDA backends needed; HIP/SYCL not planned.
- HDF5 C API (no HighFive)
- SDR → MT conversion (shared with Julia via `shared/mt/`)

## Backend Design

```cpp
// forward/src/backends/device.h
enum class Backend { OpenMP, CUDA };

// Primary template provides OpenMP backend
template <Backend B>
struct Device {
    template <typename F>
    static void parallel_for(int n, F&& f) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) f(i);
    }
};

#ifdef __CUDACC__
// CUDA specialization guarded by nvcc
struct Device<Backend::CUDA> {
    template <typename F>
    static void parallel_for(int n, F&& f) {
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        device_parallel_for_kernel<<<grid_size, block_size>>>(f, n);
    }
};
#endif
```

Kernel functions in `namespace fm` are templates parameterized on `Backend B`. Single source, compiled twice (g++ with OpenMP, nvcc for GPU). Flat arrays (`const double*`) with explicit column-major strides replace `Kokkos::View`.

**Note:** CUDA is installed via Spack on the target system. Use `spack load cuda` before building with `-DBACKEND=CUDA`.

## Implementation Phasing

1. Framework: main(), HDF5 I/O, trial reading + DataCache + misfit writing (all inline in main.cpp)
2. XCorr kernel (most complex — validates full pipeline)
3. Polarity + PSR kernels
4. AbsShift + RelShift (deferred). CAP, FFT-based XCorr — cancelled.

## What It Does NOT Do

- Does NOT apply module weights or masks
- Does NOT aggregate misfits
- Does NOT know about convergence or strategy
- Does NOT read `config.jl` directly (it consumes config already written into HDF5)
