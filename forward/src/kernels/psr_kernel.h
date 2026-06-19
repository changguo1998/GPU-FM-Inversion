#ifndef PSR_KERNEL_H
#define PSR_KERNEL_H

#include "backends/device.h"
#include <cmath>

// ─────────────────────────────────────────────────────────
// PSR (P-to-S amplitude ratio) misfit kernel
//
// Misfit formula:
//   syn_amp_P = √(mᵀ · amp_P · m)
//   syn_amp_S = √(mᵀ · amp_S · m)
//   misfit    = (log₁₀(syn_amp_P / syn_amp_S) - obs_psr)²
//
// Data layout (all flat arrays, column-major):
//   mt       [N_trials × 6]         mt[trial + comp * N_trials]
//   amp_P    [N_stations × 6 × 6]   amp_P[station + i * N_stations + j * (N_stations * 6)]
//   amp_S    [N_stations × 6 × 6]   same layout
//   obs_psr  [N_stations]           obs_psr[station]
//   misfit   [N_stations × N_trials] misfit[station + trial * N_stations]
//
// Missing stations (obs_psr is NaN, or both amp matrices near-zero):
// returns NaN.
//
// Launch: Device<B>::parallel_for over [0, N_stations * N_trials)
//         flat index = station * N_trials + trial
// ─────────────────────────────────────────────────────────

namespace fm {

/// Launch PSR misfit kernel.
template <Backend B>
void launch_psr_kernel(
    const double* mt,       // N_trials × 6, column-major: mt[trial + comp * N_trials]
    const double* amp_P,    // N_stations × 6 × 6: amp_P[station + i*S + j*(S*6)]
    const double* amp_S,    // N_stations × 6 × 6: same layout
    const double* obs_psr,  // N_stations
    double* misfit,         // N_stations × N_trials: misfit[station + trial * N_stations]
    int N_stations,
    int N_trials)
{
    Device<B>::parallel_for(N_stations * N_trials, [=] (int idx) {
        const int station = idx / N_trials;
        const int trial  = idx % N_trials;

        double obs = obs_psr[station];
        if (std::isnan(obs)) {
            misfit[station + trial * N_stations] = NAN;
            return;
        }

        // Compute quadratic forms:
        // amp_P_quad = Σᵢⱼ amp_P[station][i][j] * mt[trial][i] * mt[trial][j]
        double amp_P_quad = 0.0;
        double amp_S_quad = 0.0;
        for (int i = 0; i < 6; ++i) {
            double m_i = mt[trial + i * N_trials];
            for (int j = 0; j < 6; ++j) {
                double m_j = mt[trial + j * N_trials];
                amp_P_quad += amp_P[station + i * N_stations + j * (N_stations * 6)] * m_i * m_j;
                amp_S_quad += amp_S[station + i * N_stations + j * (N_stations * 6)] * m_i * m_j;
            }
        }

        // Numerical safety: clamp small negative values to zero
        if (amp_P_quad < 0.0) amp_P_quad = 0.0;
        if (amp_S_quad < 0.0) amp_S_quad = 0.0;

        double syn_amp_P = std::sqrt(amp_P_quad);
        double syn_amp_S = std::sqrt(amp_S_quad);

        // Skip if amplitude ratio is near-zero (degenerate)
        if (syn_amp_P < 1e-30 || syn_amp_S < 1e-30) {
            misfit[station + trial * N_stations] = NAN;
            return;
        }

        double syn_psr = std::log10(syn_amp_P / syn_amp_S);
        double diff = syn_psr - obs;
        misfit[station + trial * N_stations] = diff * diff;
    });
}

} // namespace fm

#endif // PSR_KERNEL_H