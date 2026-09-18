#include "validation.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace fm {

size_t checked_add(size_t lhs, size_t rhs, const std::string &context) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
        throw std::runtime_error(context + ": size addition overflow");
    return lhs + rhs;
}

size_t checked_mul(size_t lhs, size_t rhs, const std::string &context) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        throw std::runtime_error(context + ": size multiplication overflow");
    return lhs * rhs;
}

namespace {

void validate_index_array(const std::vector<int> &indices, size_t expected_length, size_t axis_size,
                          const char *name) {
    if (indices.size() != expected_length) {
        throw std::runtime_error(std::string("trial index length mismatch for ") + name +
                                 ": expected " + std::to_string(expected_length) + ", got " +
                                 std::to_string(indices.size()));
    }
    for (size_t trial = 0; trial < indices.size(); ++trial) {
        const int index = indices[trial];
        if (index < 1 || static_cast<size_t>(index) > axis_size) {
            throw std::runtime_error(std::string("trial ") + std::to_string(trial) +
                                     " has invalid " + name + " index " + std::to_string(index));
        }
    }
}

} // namespace

void validate_trial_indices(int n_trials, const TrialIndexView &indices,
                            const ParaspaceSizes &sizes) {
    if (n_trials <= 0)
        throw std::runtime_error("N_trials must be positive");

    const size_t count = static_cast<size_t>(n_trials);
    validate_index_array(*indices.strike, count, sizes.strike, "strike");
    validate_index_array(*indices.dip, count, sizes.dip, "dip");
    validate_index_array(*indices.rake, count, sizes.rake, "rake");
    validate_index_array(*indices.depth, count, sizes.depth, "depth");
    validate_index_array(*indices.frequency, count, sizes.frequency, "frequency");
    validate_index_array(*indices.duration, count, sizes.duration, "duration");
}

void mark_trial_completed(std::vector<uint8_t> &completed, size_t trial_index) {
    if (trial_index >= completed.size())
        throw std::runtime_error("completed trial index out of range: " +
                                 std::to_string(trial_index));
    if (completed[trial_index] != 0)
        throw std::runtime_error("trial completed more than once: " + std::to_string(trial_index));
    completed[trial_index] = 1;
}

void validate_trials_completed(const std::vector<uint8_t> &completed) {
    for (size_t trial = 0; trial < completed.size(); ++trial) {
        if (completed[trial] != 1)
            throw std::runtime_error("trial was not completed: " + std::to_string(trial));
    }
}

void validate_finite(const double *values, size_t count, const std::string &context) {
    if (values == nullptr && count != 0)
        throw std::runtime_error(context + ": null numeric input");
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            throw std::runtime_error(context + ": non-finite value at index " +
                                     std::to_string(index));
    }
}

} // namespace fm
