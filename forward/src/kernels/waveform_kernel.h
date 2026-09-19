#ifndef WAVEFORM_KERNEL_H
#define WAVEFORM_KERNEL_H

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define FM_WAVEFORM_HOST_DEVICE __host__ __device__
#else
#define FM_WAVEFORM_HOST_DEVICE
#endif

namespace fm {

/// Compute half peak-to-peak amplitude and the sign of the earlier extremum.
FM_WAVEFORM_HOST_DEVICE inline void waveform_scale_work_item(const double *mt, const double *gf,
                                                             double *amp_scale_out,
                                                             int8_t *sign_scale_out, int n_phases,
                                                             int n_trials, int n_samples,
                                                             int index) {
    const int phase = index / n_trials;
    const int trial = index % n_trials;
    const size_t output_index = static_cast<size_t>(phase) + static_cast<size_t>(trial) * n_phases;
    if (n_samples <= 0) {
        amp_scale_out[output_index] = 0.0;
        sign_scale_out[output_index] = 0;
        return;
    }

    double minimum = 0.0;
    double maximum = 0.0;
    int minimum_index = 0;
    int maximum_index = 0;
    for (int sample = 0; sample < n_samples; ++sample) {
        double value = 0.0;
        for (int component = 0; component < 6; ++component) {
            const size_t gf_index = static_cast<size_t>(phase) +
                                    static_cast<size_t>(sample) * n_phases +
                                    static_cast<size_t>(component) * n_phases * n_samples;
            value += gf[gf_index] * mt[trial * 6 + component];
        }
        if (sample == 0 || value < minimum) {
            minimum = value;
            minimum_index = sample;
        }
        if (sample == 0 || value > maximum) {
            maximum = value;
            maximum_index = sample;
        }
    }

    amp_scale_out[output_index] = (maximum - minimum) / 2.0;
    const double signed_extreme = minimum_index <= maximum_index ? minimum : maximum;
    sign_scale_out[output_index] =
        signed_extreme > 0.0 ? int8_t {1} : (signed_extreme < 0.0 ? int8_t {-1} : int8_t {0});
}

inline void launch_waveform_openmp(const double *mt, const double *gf, double *amp_scale_out,
                                   int8_t *sign_scale_out, int n_phases, int n_trials,
                                   int n_samples) {
#pragma omp parallel for
    for (int index = 0; index < n_phases * n_trials; ++index) {
        waveform_scale_work_item(mt, gf, amp_scale_out, sign_scale_out, n_phases, n_trials,
                                 n_samples, index);
    }
}

void launch_waveform_cuda(const double *mt, const double *gf, double *amp_scale_out,
                          int8_t *sign_scale_out, int n_phases, int n_trials, int n_samples);

} // namespace fm

#undef FM_WAVEFORM_HOST_DEVICE

#endif // WAVEFORM_KERNEL_H
