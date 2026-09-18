#ifndef FM_CUDA_BATCH_H
#define FM_CUDA_BATCH_H

#include <cstddef>
#include <optional>

namespace fm {

struct CudaBatchPlan {
    size_t capacity;
    size_t per_trial_bytes;
    size_t safety_margin;
};

/// Plan a positive CUDA trial batch from post-combo free memory.
CudaBatchPlan plan_cuda_batch(size_t free_bytes, size_t total_bytes, size_t n_phases,
                              size_t n_trials, std::optional<size_t> explicit_limit);

} // namespace fm

#endif // FM_CUDA_BATCH_H
