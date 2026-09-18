#ifndef FM_VALIDATION_H
#define FM_VALIDATION_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fm {

struct TrialIndexView {
    const std::vector<int> *strike;
    const std::vector<int> *dip;
    const std::vector<int> *rake;
    const std::vector<int> *depth;
    const std::vector<int> *frequency;
    const std::vector<int> *duration;

    TrialIndexView(const std::vector<int> &strike_in, const std::vector<int> &dip_in,
                   const std::vector<int> &rake_in, const std::vector<int> &depth_in,
                   const std::vector<int> &frequency_in, const std::vector<int> &duration_in)
        : strike(&strike_in), dip(&dip_in), rake(&rake_in), depth(&depth_in),
          frequency(&frequency_in), duration(&duration_in) {
    }
};

struct ParaspaceSizes {
    size_t strike;
    size_t dip;
    size_t rake;
    size_t depth;
    size_t frequency;
    size_t duration;
};

/// Add two sizes and throw on overflow.
size_t checked_add(size_t lhs, size_t rhs, const std::string &context);

/// Multiply two sizes and throw on overflow.
size_t checked_mul(size_t lhs, size_t rhs, const std::string &context);

/// Validate all trial index arrays and their 1-based paraspace bounds.
void validate_trial_indices(int n_trials, const TrialIndexView &indices,
                            const ParaspaceSizes &sizes);

/// Mark one trial complete, rejecting out-of-range and duplicate marks.
void mark_trial_completed(std::vector<uint8_t> &completed, size_t trial_index);

/// Require every trial to have been completed exactly once.
void validate_trials_completed(const std::vector<uint8_t> &completed);

/// Reject NaN and Inf values in an active numeric input.
void validate_finite(const double *values, size_t count, const std::string &context);

} // namespace fm

#endif // FM_VALIDATION_H
