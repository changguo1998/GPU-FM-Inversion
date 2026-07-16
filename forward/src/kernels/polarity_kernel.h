#ifndef POLARITY_KERNEL_H
#define POLARITY_KERNEL_H

#include "backends/device.h"
#include <cmath>

// ─────────────────────────────────────────────────────────
// Polarity kernel
//
// Outputs (intermediate products, NOT final misfit):
//   syn_sign[station,trial]  = sign(Σᵢ pol_vec[station][i] * mt[trial][i])  (Int8: -1/0/1)
//   dot_value[station,trial] = Σᵢ pol_vec[station][i] * mt[trial][i]        (Float64, raw dot)
//
// Misfit (syn_sign mismatch) is derived in Julia assess:
//   (syn_sign == obs_pol) ? 0.0 : 1.0
//
// Data layout (all flat arrays, column-major):
//   mt            [N_trials × 6]      mt[trial + comp * N_trials]
//   pol_vec       [N_stations × 6]    pol_vec[station + comp * N_stations]
//   syn_sign_out  [N_stations × N_trials]  column-major
//   dot_value_out [N_stations × N_trials]  column-major
// ─────────────────────────────────────────────────────────

namespace fm {

/// Launch polarity kernel. Outputs syn_sign + dot_value intermediates.
template <Backend B>
void launch_polarity_kernel(const double *mt,      // N_trials × 6, column-major
                            const double *pol_vec, // N_stations × 6, column-major
                            int8_t *syn_sign_out,  // N_stations × N_trials
                            double *dot_value_out, // N_stations × N_trials
                            int N_stations, int N_trials) {
    Device<B>::parallel_for(N_stations * N_trials, [=](int idx) {
        const int station = idx / N_trials;
        const int trial = idx % N_trials;

        double dot = 0.0;
        for (int c = 0; c < 6; ++c) {
            dot += pol_vec[station + c * N_stations] * mt[trial + c * N_trials];
        }

        int syn = (dot > 0.0) ? 1 : ((dot < 0.0) ? -1 : 0);
        syn_sign_out[station + trial * N_stations] = static_cast<int8_t>(syn);
        dot_value_out[station + trial * N_stations] = dot;
    });
}

} // namespace fm

#endif // POLARITY_KERNEL_H
