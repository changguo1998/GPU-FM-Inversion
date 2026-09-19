#include "backends/cuda_runtime.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<double> moment_tensors(size_t n_trials) {
    std::vector<double> mt(n_trials * 6, 0.0);
    for (size_t trial = 0; trial < n_trials; ++trial)
        mt[trial * 6] = 1.0;
    return mt;
}

void evaluate_case(fm::CudaXcorrExecutor &executor, const std::vector<double> &cc_values,
                   const std::vector<double> &syn_norm2, double obs_norm2, size_t n_trials,
                   double expected_cc, int32_t expected_lag, const std::string &context) {
    const size_t cc_rows = cc_values.size();
    require(cc_rows == syn_norm2.size(), context + ": invalid test fixture");

    std::vector<double> cc(cc_rows * 6, 0.0);
    std::vector<double> synamp(cc_rows * 36, 0.0);
    for (size_t lag = 0; lag < cc_rows; ++lag) {
        cc[lag] = cc_values[lag];
        synamp[lag * 36] = syn_norm2[lag];
    }

    const std::vector<double> mt = moment_tensors(n_trials);
    const std::vector<double> gf = {-4.0, 2.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> output(n_trials, 123.0);
    std::vector<int32_t> lag(n_trials, 123);
    std::vector<double> energy(n_trials, -1.0);
    std::vector<double> amplitude(n_trials, -1.0);
    std::vector<int8_t> sign(n_trials, 9);
    executor.evaluate(mt.data(), cc.data(), synamp.data(), &obs_norm2, gf.data(), output.data(),
                      lag.data(), energy.data(), amplitude.data(), sign.data(), n_trials, cc_rows,
                      4, static_cast<int>(cc_rows / 2), context);
    for (size_t trial = 0; trial < n_trials; ++trial) {
        require(output[trial] == expected_cc, context + ": wrong CC");
        require(lag[trial] == expected_lag, context + ": wrong lag");
        require(energy[trial] == syn_norm2[cc_rows / 2], context + ": wrong energy");
        require(amplitude[trial] == 3.5, context + ": wrong amplitude");
        require(sign[trial] == -1, context + ": wrong sign");
    }
}

} // namespace

int main() {
    const fm::CudaProbeResult probe = fm::probe_cuda_device();
    require(probe.status == fm::CudaProbeStatus::Available,
            "CUDA device unavailable: " + probe.error);

    fm::CudaXcorrExecutor executor(1, 3, 4, 5, 2, true, true);
    evaluate_case(executor, {0.5, 0.5, 0.4}, {1.0, 1.0, 1.0}, 1.0, 5, 0.5, -1,
                  "exact tie residual batch");

    const double above_half = std::nextafter(0.5, 1.0);
    evaluate_case(executor, {0.5, above_half, 0.4}, {1.0, 1.0, 1.0}, 1.0, 1, above_half, 0,
                  "near tie smaller combo");
    evaluate_case(executor, {-0.5, -0.2, -0.3}, {1.0, 1.0, 1.0}, 1.0, 3, -0.2, 0,
                  "signed negative correlation");
    evaluate_case(executor, {-0.75}, {1.0}, 1.0, 2, -0.75, 0, "single-row overwrite");
    evaluate_case(executor, {0.5, 0.6, 0.7}, {1.0, 1.0, 1.0}, 0.0, 2, 0.0, 0,
                  "zero observation norm");
    evaluate_case(executor, {0.5, 0.6, 0.7}, {0.0, 0.0, 0.0}, 1.0, 2, 0.0, 0,
                  "degenerate synthetic norm");

    std::cout << "xcorr_cuda_test: PASS" << std::endl;
    return 0;
}
