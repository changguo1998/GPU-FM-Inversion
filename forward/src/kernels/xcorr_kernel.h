#ifndef XCORR_KERNEL_H
#define XCORR_KERNEL_H

/// XCorr misfit kernel - time-domain cross-correlation.
///
/// Operates on precomputed CC(obs, GF[:,i]) and synamp data.
/// One parallel work-item per (phase × trial) combination.
///
/// Outputs (intermediate products, NOT final misfit):
///   cc_max[phase,trial]   = max_k cc_norm[k]               (Float64)
///   best_lag[phase,trial] = argmax_k cc_norm[k] - maxlag   (Int32, samples)
///   energy[phase,trial]   = mᵀ Gram[k=0] m                 (Float64, optional)
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

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define FM_XCORR_HOST_DEVICE __host__ __device__
#else
#define FM_XCORR_HOST_DEVICE
#endif

namespace fm {

/// Evaluate one (phase, trial) XCorr work item on either host or device.
FM_XCORR_HOST_DEVICE inline void
xcorr_work_item(const double *mt,          // N_trials × 6, row-major: mt[trial * 6 + comp]
                const double *cc_data,     // [N_phases·cc_pp × 6] column-major
                const double *synamp_data, // [N_phases × 36 × cc_pp] column-major (per-lag Gram)
                const double *obs_norm2,   // [N_phases]
                double *cc_max_out,        // [N_phases × N_trials] column-major
                int32_t *best_lag_out,     // [N_phases × N_trials] column-major
                double *energy_out,        // optional [N_phases × N_trials] column-major
                int N_phases, int N_trials, int cc_pp, int maxlag, int idx) {
    const int phase = idx / N_trials;
    const int trial = idx % N_trials;

    // ── Load moment tensor (6-comp) for this trial ──
    double m[6];
    for (int c = 0; c < 6; ++c) {
        m[c] = mt[trial * 6 + c];
    }

    // Precompute the 21 unique coefficients of mᵀAm once per work item.
    double quadratic_coeffs[21];
    int coeff = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            quadratic_coeffs[coeff++] = (i == j ? 1.0 : 2.0) * m[i] * m[j];
        }
    }

    // ── cc_syn[k] = Σᵢ m[i] · CC[phase][k][i], track max + argmax ──
    // Per-lag normalization: syn_norm²[k] = mᵀ · synamp[:,:,k] · m
    const size_t cc_start = static_cast<size_t>(phase) * static_cast<size_t>(cc_pp);
    const size_t syn_stride = static_cast<size_t>(N_phases) * 36;
    const size_t cc_stride = static_cast<size_t>(N_phases) * static_cast<size_t>(cc_pp);
    const size_t output_index =
        static_cast<size_t>(phase) + static_cast<size_t>(trial) * static_cast<size_t>(N_phases);

    if (energy_out != nullptr) {
        double energy = 0.0;
        const double *synamp_zero = synamp_data + static_cast<size_t>(maxlag) * syn_stride;
        coeff = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                const size_t offset =
                    static_cast<size_t>(phase) +
                    static_cast<size_t>(i * 6 + j) * static_cast<size_t>(N_phases);
                energy += quadratic_coeffs[coeff++] * synamp_zero[offset];
            }
        }
        energy_out[output_index] = energy > 0.0 ? energy : 0.0;
    }

    // Guard: zero obs norm -> no correlation information, but energy remains valid.
    const double obs_n2 = obs_norm2[phase];
    if (obs_n2 <= 0.0) {
        cc_max_out[output_index] = 0.0;
        best_lag_out[output_index] = 0;
        return;
    }

    double best_cc = 0.0;
    bool found_cc = false;
    int best_k = maxlag; // default zero-shift
    for (int k = 0; k < cc_pp; ++k) {
        double syn_norm2 = 0.0;
        const double *synamp_k = synamp_data + static_cast<size_t>(k) * syn_stride;
        coeff = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                const size_t offset =
                    static_cast<size_t>(phase) +
                    static_cast<size_t>(i * 6 + j) * static_cast<size_t>(N_phases);
                syn_norm2 += quadratic_coeffs[coeff++] * synamp_k[offset];
            }
        }
        if (syn_norm2 <= 0.0) {
            continue; // degenerate lag
        }
        const double denom = ::sqrt(obs_n2 * syn_norm2);
        double cc_syn = 0.0;
        for (int i = 0; i < 6; ++i) {
            cc_syn +=
                m[i] *
                cc_data[cc_start + static_cast<size_t>(k) + static_cast<size_t>(i) * cc_stride];
        }
        double cc_norm = cc_syn / denom;
        if (!found_cc || cc_norm > best_cc) {
            best_cc = cc_norm;
            best_k = k;
            found_cc = true;
        }
    }

    cc_max_out[output_index] = best_cc;
    best_lag_out[output_index] = static_cast<int32_t>(best_k - maxlag);
}

/// Launch the XCorr work items with OpenMP.
inline void launch_xcorr_openmp(const double *mt, const double *cc_data, const double *synamp_data,
                                const double *obs_norm2, double *cc_max_out, int32_t *best_lag_out,
                                double *energy_out, int N_phases, int N_trials, int cc_pp,
                                int maxlag) {
#pragma omp parallel for
    for (int idx = 0; idx < N_phases * N_trials; ++idx) {
        xcorr_work_item(mt, cc_data, synamp_data, obs_norm2, cc_max_out, best_lag_out, energy_out,
                        N_phases, N_trials, cc_pp, maxlag, idx);
    }
}

/// Launch XCorr work items on CUDA device buffers.
void launch_xcorr_cuda(const double *mt, const double *cc_data, const double *synamp_data,
                       const double *obs_norm2, double *cc_max_out, int32_t *best_lag_out,
                       double *energy_out, int n_phases, int n_trials, int cc_pp, int maxlag);

} // namespace fm

#undef FM_XCORR_HOST_DEVICE

#endif // XCORR_KERNEL_H
