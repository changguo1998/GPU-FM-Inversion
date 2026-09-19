#include "backends/cuda_batch.h"

#include "validation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fm {

CudaBatchPlan plan_cuda_batch(size_t free_bytes, size_t total_bytes, size_t n_phases,
                              size_t n_trials, std::optional<size_t> explicit_limit,
                              size_t extra_per_trial_bytes) {
    if (n_phases == 0 || n_trials == 0)
        throw std::runtime_error("CUDA batch planning requires phases and trials");
    if (explicit_limit && *explicit_limit == 0)
        throw std::runtime_error("CUDA batch limit must be positive");

    constexpr size_t MINIMUM_MARGIN = 256ULL * 1024ULL * 1024ULL;
    const size_t safety_margin = std::max(MINIMUM_MARGIN, total_bytes / 20);
    if (free_bytes <= safety_margin)
        throw std::runtime_error("insufficient CUDA memory after safety margin");

    const size_t mt_bytes = checked_mul(6, sizeof(double), "CUDA batch MT bytes");
    const size_t cc_bytes = checked_mul(n_phases, sizeof(double), "CUDA batch CC bytes");
    const size_t lag_bytes = checked_mul(n_phases, sizeof(int32_t), "CUDA batch lag bytes");
    const size_t per_trial_bytes =
        checked_add(checked_add(checked_add(mt_bytes, cc_bytes, "CUDA batch bytes"), lag_bytes,
                                "CUDA batch bytes"),
                    extra_per_trial_bytes, "CUDA batch extra bytes");

    size_t capacity = (free_bytes - safety_margin) / per_trial_bytes;
    capacity = std::min(capacity, n_trials);
    if (explicit_limit)
        capacity = std::min(capacity, *explicit_limit);

    const size_t max_by_work_items =
        static_cast<size_t>(std::numeric_limits<int>::max()) / n_phases;
    capacity = std::min(capacity, max_by_work_items);
    if (capacity == 0)
        throw std::runtime_error("CUDA memory cannot hold one trial batch");

    return {capacity, per_trial_bytes, safety_margin};
}

} // namespace fm
