#include "kernels/waveform_kernel.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fm {
namespace {

__global__ void waveform_kernel(const double *mt, const double *gf, double *amp_scale_out,
                                int8_t *sign_scale_out, int n_phases, int n_trials, int n_samples,
                                int work_items) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < work_items)
        waveform_scale_work_item(mt, gf, amp_scale_out, sign_scale_out, n_phases, n_trials,
                                 n_samples, index);
}

} // namespace

void launch_waveform_cuda(const double *mt, const double *gf, double *amp_scale_out,
                          int8_t *sign_scale_out, int n_phases, int n_trials, int n_samples) {
    constexpr int THREADS_PER_BLOCK = 128;
    const int work_items = n_phases * n_trials;
    const int blocks = (work_items - 1) / THREADS_PER_BLOCK + 1;
    waveform_kernel<<<blocks, THREADS_PER_BLOCK>>>(mt, gf, amp_scale_out, sign_scale_out, n_phases,
                                                   n_trials, n_samples, work_items);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA waveform launch failed: ") +
                                 cudaGetErrorString(error));
}

} // namespace fm
