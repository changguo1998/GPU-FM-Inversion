#include "kernels/xcorr_kernel.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

struct Result {
    double cc;
    int32_t lag;
};

Result evaluate(const std::vector<double> &cc_values, double obs_norm2,
                const std::vector<double> &syn_norm2) {
    const int cc_rows = static_cast<int>(cc_values.size());
    std::vector<double> cc(static_cast<size_t>(cc_rows) * 6, 0.0);
    std::vector<double> synamp(static_cast<size_t>(cc_rows) * 36, 0.0);
    for (int lag = 0; lag < cc_rows; ++lag) {
        cc[static_cast<size_t>(lag)] = cc_values[static_cast<size_t>(lag)];
        synamp[static_cast<size_t>(lag) * 36] = syn_norm2[static_cast<size_t>(lag)];
    }
    const double mt[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double output = 0.0;
    int32_t lag = 0;
    fm::xcorr_work_item(mt, cc.data(), synamp.data(), &obs_norm2, &output, &lag, 1, 1, cc_rows,
                        cc_rows / 2, 0);
    return {output, lag};
}

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    const Result tie = evaluate({0.5, 0.5, 0.4}, 1.0, {1.0, 1.0, 1.0});
    require(tie.cc == 0.5 && tie.lag == -1, "exact tie must select first valid lag");

    const double above_half = std::nextafter(0.5, 1.0);
    const Result near_tie = evaluate({0.5, above_half, 0.4}, 1.0, {1.0, 1.0, 1.0});
    require(near_tie.cc == above_half && near_tie.lag == 0,
            "near tie must select larger signed CC");

    const Result negative = evaluate({-0.5, -0.2, -0.3}, 1.0, {1.0, 1.0, 1.0});
    require(negative.cc == -0.2 && negative.lag == 0, "negative CC must not use absolute value");

    const Result zero_observation = evaluate({0.5, 0.6, 0.7}, 0.0, {1.0, 1.0, 1.0});
    require(zero_observation.cc == 0.0 && zero_observation.lag == 0,
            "zero observation norm must be neutral");

    const Result degenerate = evaluate({0.5, 0.6, 0.7}, 1.0, {0.0, 0.0, 0.0});
    require(degenerate.cc == 0.0 && degenerate.lag == 0,
            "degenerate synthetic norm must be neutral");
    return 0;
}
