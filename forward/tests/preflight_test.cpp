#include "validation.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename F> void expect_throw(const std::string &name, F &&f) {
    try {
        f();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("expected failure: " + name);
}

fm::TrialIndexView valid_indices() {
    static const std::vector<int> strike {1, 2};
    static const std::vector<int> dip {1, 1};
    static const std::vector<int> rake {2, 1};
    static const std::vector<int> depth {1, 2};
    static const std::vector<int> frequency {1, 1};
    static const std::vector<int> duration {2, 1};
    return {strike, dip, rake, depth, frequency, duration};
}

} // namespace

int main() {
    if (fm::checked_add(2, 3, "sum") != 5 || fm::checked_mul(3, 4, "product") != 12)
        return 1;
    expect_throw("checked_add overflow",
                 [] { fm::checked_add(std::numeric_limits<size_t>::max(), 1, "sum"); });
    expect_throw("checked_mul overflow",
                 [] { fm::checked_mul(std::numeric_limits<size_t>::max(), 2, "product"); });

    const fm::ParaspaceSizes sizes {2, 1, 2, 2, 1, 2};
    fm::validate_trial_indices(2, valid_indices(), sizes);
    expect_throw("zero trials", [&] { fm::validate_trial_indices(0, valid_indices(), sizes); });

    std::vector<int> short_strike {1};
    auto indices = valid_indices();
    indices.strike = &short_strike;
    expect_throw("length mismatch", [&] { fm::validate_trial_indices(2, indices, sizes); });

    using IndexMember = const std::vector<int> *fm::TrialIndexView::*;
    const std::vector<IndexMember> index_members {
        &fm::TrialIndexView::strike, &fm::TrialIndexView::dip,       &fm::TrialIndexView::rake,
        &fm::TrialIndexView::depth,  &fm::TrialIndexView::frequency, &fm::TrialIndexView::duration,
    };
    std::vector<int> invalid_zero {0, 1};
    for (IndexMember member : index_members) {
        indices = valid_indices();
        indices.*member = &invalid_zero;
        expect_throw("zero index", [&] { fm::validate_trial_indices(2, indices, sizes); });
    }

    std::vector<int> invalid_depth {1, 3};
    indices = valid_indices();
    indices.depth = &invalid_depth;
    expect_throw("index out of range", [&] { fm::validate_trial_indices(2, indices, sizes); });

    std::vector<uint8_t> completed(2, 0);
    fm::mark_trial_completed(completed, 1);
    expect_throw("duplicate completion", [&] { fm::mark_trial_completed(completed, 1); });
    expect_throw("incomplete trials", [&] { fm::validate_trials_completed(completed); });
    fm::mark_trial_completed(completed, 0);
    fm::validate_trials_completed(completed);

    const double finite[] = {0.0, -1.0, 2.0};
    fm::validate_finite(finite, 3, "finite");
    const double nan_values[] = {std::numeric_limits<double>::quiet_NaN()};
    expect_throw("NaN input", [&] { fm::validate_finite(nan_values, 1, "nan"); });
    const double inf_values[] = {std::numeric_limits<double>::infinity()};
    expect_throw("Inf input", [&] { fm::validate_finite(inf_values, 1, "inf"); });

    std::cout << "preflight_test: PASS" << std::endl;
    return 0;
}
