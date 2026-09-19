#include "kernels/waveform_kernel.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    constexpr int n_phases = 2;
    constexpr int n_trials = 2;
    constexpr int n_samples = 4;

    std::vector<double> gf(n_phases * n_samples * 6, 0.0);
    const double phase0[] = {-4.0, 2.0, 3.0, -1.0};
    const double phase1[] = {0.0, -2.0, 2.0, 0.0};
    for (int sample = 0; sample < n_samples; ++sample) {
        gf[0 + sample * n_phases] = phase0[sample];
        gf[1 + sample * n_phases] = phase1[sample];
    }

    const double mt[] = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    std::vector<double> amplitude(n_phases * n_trials, -1.0);
    std::vector<int8_t> sign(n_phases * n_trials, 9);
    fm::launch_waveform_openmp(mt, gf.data(), amplitude.data(), sign.data(), n_phases, n_trials,
                               n_samples);

    require(amplitude[0] == 3.5 && sign[0] == -1, "phase 0 trial 0 mismatch");
    require(amplitude[1] == 2.0 && sign[1] == -1, "equal extrema must use the earlier sample");
    require(amplitude[2] == 7.0 && sign[2] == 1, "phase 0 trial 1 mismatch");
    require(amplitude[3] == 4.0 && sign[3] == 1, "phase 1 trial 1 mismatch");

    double zero_amplitude = -1.0;
    int8_t zero_sign = 9;
    fm::waveform_scale_work_item(mt, gf.data(), &zero_amplitude, &zero_sign, 1, 1, 0, 0);
    require(zero_amplitude == 0.0 && zero_sign == 0, "empty waveform must be neutral");
    return 0;
}
