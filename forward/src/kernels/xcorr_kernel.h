#ifndef XCORR_KERNEL_H
#define XCORR_KERNEL_H

/// XCorr misfit kernel - time-domain cross-correlation.
///
/// Operates on precomputed CC(obs, GF[:,i]) and synamp data.
/// One parallel work-item per (phase × trial) combination.
///
/// Outputs (intermediate products, NOT final misfit):
///   cc_max[phase,trial]   = max_k |cc_norm[k]|        (Float64)
///   best_lag[phase,trial] = argmax_k - maxlag          (Int32, relative shift in samples)
///
/// Misfit (cc_misfit) is derived in Julia assess: 1.0 - cc_max.
/// AbsShift is derived in Julia assess: best_lag * dt.
///
/// Data layout (all flat arrays, column-major):
///   mt           [N_trials × 6]       mt[trial * 6 + comp] (row-major)
///   cc_data      [N_phases·cc_pp × 6] column-major
///   synamp_data  [N_phases × 36]      synamp_data[phase + (i*6+j) * N_phases]
///   obs_norm2    [N_phases]           obs_norm2[phase]
///   cc_max_out   [N_phases × N_trials] column-major
///   best_lag_out [N_phases × N_trials] column-major
///
/// Verification invariant: mᵀ·synamp·m = ‖GF·m‖²  (Gram matrix identity)

#include "backends/device.h"
#include <cmath>

namespace fm {

/// Launch the XCorr kernel over (N_phases × N_trials) flat work items.
template <Backend B>
inline void launch_xcorr_misfit(
    const double *mt,          // N_trials × 6, row-major: mt[trial * 6 + comp]
    const double *cc_data,     // [N_phases·cc_pp × 6] column-major
    const double *synamp_data, // [N_phases × 36 × cc_pp] column-major (per-lag Gram)
    const double *obs_norm2,   // [N_phases]
    double *cc_max_out,        // [N_phases × N_trials] column-major
    int32_t *best_lag_out,     // [N_phases × N_trials] column-major
    int N_phases, int N_trials, int cc_pp, int maxlag) {
    Device<B>::parallel_for(N_phases * N_trials, [=](int idx) {
        const int phase = idx / N_trials;
        const int trial = idx % N_trials;

        // ── Load moment tensor (6-comp) for this trial ──
        double m[6];
        for (int c = 0; c < 6; ++c) {
            m[c] = mt[trial * 6 + c];
        }

        // ── Work items processed per (phase, trial) ──
        const double obs_n2 = obs_norm2[phase];

        // Guard: zero obs norm -> no information
        if (obs_n2 <= 0.0) {
            cc_max_out[phase + trial * N_phases] = 0.0;
            best_lag_out[phase + trial * N_phases] = 0;
            return;
        }

        // ── cc_syn[k] = Σᵢ m[i] · CC[phase][k][i], track max + argmax ──
        // Per-lag normalization: syn_norm²[k] = mᵀ · synamp[:,:,k] · m
        const int cc_start = phase * cc_pp;
        const int syn_stride = N_phases * 36;
        double max_abs_cc = 0.0;
        int best_k = maxlag; // default zero-shift
        for (int k = 0; k < cc_pp; ++k) {
            double syn_norm2 = 0.0;
            const double *synamp_k = synamp_data + k * syn_stride;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    syn_norm2 += m[i] * synamp_k[phase + (i * 6 + j) * N_phases] * m[j];
                }
            }
            if (syn_norm2 <= 0.0) {
                continue; // degenerate lag
            }
            const double denom = std::sqrt(obs_n2 * syn_norm2);
            double cc_syn = 0.0;
            for (int i = 0; i < 6; ++i) {
                cc_syn += m[i] * cc_data[(cc_start + k) + i * (N_phases * cc_pp)];
            }
            double cc_norm = cc_syn / denom;
            double abs_cc = std::fabs(cc_norm);
            if (abs_cc > max_abs_cc) {
                max_abs_cc = abs_cc;
                best_k = k;
            }
        }

        cc_max_out[phase + trial * N_phases] = max_abs_cc;
        best_lag_out[phase + trial * N_phases] = static_cast<int32_t>(best_k - maxlag);
    });
}

} // namespace fm

#endif // XCORR_KERNEL_H
