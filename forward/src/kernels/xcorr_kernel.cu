#include "kernels/xcorr_kernel.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fm {
namespace {

__global__ void xcorr_kernel(const double *mt, const double *cc_data, const double *synamp_data,
                             const double *obs_norm2, double *cc_max_out, int32_t *best_lag_out,
                             int n_phases, int n_trials, int cc_pp, int maxlag, int work_items) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < work_items)
        xcorr_work_item(mt, cc_data, synamp_data, obs_norm2, cc_max_out, best_lag_out, n_phases,
                        n_trials, cc_pp, maxlag, index);
}

} // namespace

void launch_xcorr_cuda(const double *mt, const double *cc_data, const double *synamp_data,
                       const double *obs_norm2, double *cc_max_out, int32_t *best_lag_out,
                       int n_phases, int n_trials, int cc_pp, int maxlag) {
    constexpr int THREADS_PER_BLOCK = 128;
    const int work_items = n_phases * n_trials;
    const int blocks = (work_items - 1) / THREADS_PER_BLOCK + 1;
    xcorr_kernel<<<blocks, THREADS_PER_BLOCK>>>(mt, cc_data, synamp_data, obs_norm2, cc_max_out,
                                                best_lag_out, n_phases, n_trials, cc_pp, maxlag,
                                                work_items);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA XCorr launch failed: ") +
                                 cudaGetErrorString(error));
}

} // namespace fm
