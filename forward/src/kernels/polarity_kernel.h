#ifndef POLARITY_KERNEL_H
#define POLARITY_KERNEL_H

#include "backends/device.h"
#include <cmath>

// ─────────────────────────────────────────────────────────
// Polarity misfit kernel
//
// Misfit formula:
//   syn_pol = sign(Σᵢ pol_vec[station][i] * mt[trial][i])
//   misfit  = (syn_pol == obs_pol) ? 0.0 : 1.0
//
// Data layout (all flat arrays, column-major):
//   mt       [N_trials × 6]      mt[trial + comp * N_trials]
//   pol_vec  [N_stations × 6]    pol_vec[station + comp * N_stations]
//   obs_pol  [N_stations]        obs_pol[station]
//   misfit   [N_stations × N_trials]  misfit[station + trial * N_stations]
//
// Missing polarity (obs_pol == 0.0 with pol_vec all-zero, or NaN):
// returns NaN to signal "not applicable".
//
// Launch: Device<B>::parallel_for over [0, N_stations * N_trials)
//         flat index = station * N_trials + trial
// ─────────────────────────────────────────────────────────

namespace fm {

/// Launch polarity misfit kernel.
template <Backend B>
void launch_polarity_kernel(
    const double *mt, // N_trials × 6, column-major: mt[trial + comp * N_trials]
    const double *pol_vec, // N_stations × 6, column-major: pol_vec[station +
                           // comp * N_stations]
    const double *obs_pol, // N_stations
    double
        *misfit, // N_stations × N_trials: misfit[station + trial * N_stations]
    int N_stations, int N_trials) {
  Device<B>::parallel_for(N_stations * N_trials, [=](int idx) {
    const int station = idx / N_trials;
    const int trial = idx % N_trials;

    double obs = obs_pol[station];

    // Check for missing polarity — NaN sentinel
    if (std::isnan(obs)) {
      misfit[station + trial * N_stations] = NAN;
      return;
    }

    // Compute dot product: Σᵢ pol_vec[station][i] * mt[trial][i]
    double dot = 0.0;
    for (int c = 0; c < 6; ++c) {
      dot += pol_vec[station + c * N_stations] * mt[trial + c * N_trials];
    }

    // Determine synthetic polarity sign
    int syn_pol = (dot > 0.0) ? 1 : ((dot < 0.0) ? -1 : 0);

    // Zero obs_pol with zero pol_vec → skip (not applicable)
    if (obs == 0.0 && syn_pol == 0) {
      misfit[station + trial * N_stations] = NAN;
      return;
    }

    int obs_int = (obs > 0.5) ? 1 : ((obs < -0.5) ? -1 : 0);

    // Match: sign matches
    misfit[station + trial * N_stations] = (syn_pol == obs_int) ? 0.0 : 1.0;
  });
}

} // namespace fm

#endif // POLARITY_KERNEL_H