#ifndef XCORR_KERNEL_H
#define XCORR_KERNEL_H

/// XCorr misfit kernel — time-domain cross-correlation misfit.
///
/// Operates on precomputed CC(obs, GF[:,i]) and synamp data.
/// One parallel work-item per (phase × trial) combination.
///
/// Misfit formula:
///   cc_syn[k]   = Σᵢ m[i] · CC[phase][k][i]
///   syn_norm²   = mᵀ · synamp · m
///   cc_norm[k]  = cc_syn[k] / √(obs_norm² · syn_norm²)
///   misfit      = 1.0 - maxₖ(|cc_norm[k]|)
///
/// Data layout (all flat arrays, column-major):
///   mt           [N_trials × 6]       mt[trial + comp * N_trials]
///   cc_data      [N_phases·cc_pp × 6] column-major
///   synamp_data  [N_phases × 36]      synamp_data[phase + (i*6+j) * N_phases]
///   obs_norm2    [N_phases]           obs_norm2[phase]
///   misfit       [N_phases × N_trials] misfit[phase + trial * N_phases]
///
/// Verification invariant: mᵀ·synamp·m = ‖GF·m‖²  (Gram matrix identity)

#include "backends/device.h"
#include <cmath>

namespace fm {

/// Launch the XCorr misfit kernel over (N_phases × N_trials) flat work items.
///
/// @tparam B           Backend (OpenMP or CUDA)
/// @param mt           Moment tensor data [N_trials × 6], column-major
/// @param cc_data      Precomputed CC(obs, GF[:,i])  [N_phases·cc_pp × 6]
/// @param synamp_data  GF auto-correlation per phase [N_phases × 36]
/// @param obs_norm2    ‖obs‖² per phase             [N_phases]
/// @param misfit       Output misfit per (phase,trial) [N_phases × N_trials]
/// @param N_phases     Number of phases
/// @param N_trials     Number of trials
/// @param cc_pp        CC lags per phase (= 2·maxlag + 1)
template <Backend B>
inline void launch_xcorr_misfit(
    const double *mt, // N_trials × 6, column-major: mt[trial + comp * N_trials]
    const double *cc_data,     // [N_phases·cc_pp × 6] column-major
    const double *synamp_data, // [N_phases × 36] column-major
    const double *obs_norm2,   // [N_phases]
    double *misfit,            // [N_phases × N_trials] column-major
    int N_phases, int N_trials, int cc_pp) {
  Device<B>::parallel_for(N_phases * N_trials, [=](int idx) {
    const int phase = idx / N_trials;
    const int trial = idx % N_trials;

    // ── Load moment tensor (6-comp) for this trial (column-major [N_trials×6])
    // ──
    double m[6];
    for (int c = 0; c < 6; ++c) {
      m[c] = mt[trial + c * N_trials];
    }

    // ── syn_norm² = mᵀ · synamp · m  (6×6 quadratic form) ──
    double syn_norm2 = 0.0;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        syn_norm2 += m[i] * synamp_data[phase + (i * 6 + j) * N_phases] * m[j];
      }
    }

    const double obs_n2 = obs_norm2[phase];

    // Guard: zero or negative norms → worst misfit (no information)
    if (syn_norm2 <= 0.0 || obs_n2 <= 0.0) {
      misfit[phase + trial * N_phases] = 1.0;
      return;
    }

    const double denom = std::sqrt(obs_n2 * syn_norm2);

    // ── cc_syn[k] = Σᵢ m[i] · CC[phase][k][i], then max-normalize ──
    const int cc_start = phase * cc_pp;
    double max_abs_cc = 0.0;

    for (int k = 0; k < cc_pp; ++k) {
      double cc_syn = 0.0;
      for (int i = 0; i < 6; ++i) {
        cc_syn += m[i] * cc_data[(cc_start + k) + i * (N_phases * cc_pp)];
      }
      double cc_norm = cc_syn / denom;
      double abs_cc = std::fabs(cc_norm);
      if (abs_cc > max_abs_cc) {
        max_abs_cc = abs_cc;
      }
    }

    misfit[phase + trial * N_phases] = 1.0 - max_abs_cc;
  });
}

} // namespace fm

#endif // XCORR_KERNEL_H