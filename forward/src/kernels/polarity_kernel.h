#ifndef POLARITY_KERNEL_H
#define POLARITY_KERNEL_H

#include <Kokkos_Core.hpp>

// ─────────────────────────────────────────────────────────
// Polarity misfit kernel
//
// Misfit formula:
//   syn_pol = sign(Σᵢ pol_vec[station][i] * mt[i][trial])
//   misfit  = (syn_pol == obs_pol) ? 0.0 : 1.0
//
// Inputs:
//   mt       [N_trials × 6]        — moment tensor components per trial
//   pol_vec  [N_stations × 6]      — summed GF over polarity window
//   obs_pol  [N_stations]          — observed polarity (-1, 0, +1, or missing NaN)
//
// Output:
//   misfit   [N_stations × N_trials] — 0.0 for match, 1.0 for mismatch
//
// Missing polarity (obs_pol == 0.0 with pol_vec all-zero, or NaN):
// returns NaN to signal "not applicable".
//
// Launch: Kokkos::RangePolicy over [0, N_stations * N_trials)
//         flat index = station * N_trials + trial
// ─────────────────────────────────────────────────────────

namespace fm {

template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
void launch_polarity_kernel(
    const Kokkos::View<const double**, Kokkos::LayoutLeft, ExecSpace>& mt,       // N_trials × 6
    const Kokkos::View<const double**, Kokkos::LayoutLeft, ExecSpace>& pol_vec,  // N_stations × 6
    const Kokkos::View<const double*,  ExecSpace>& obs_pol,                      // N_stations
    Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>& misfit)               // N_stations × N_trials
{
    const int N_stations = pol_vec.extent(0);
    const int N_trials   = mt.extent(0);
    const int N_total = N_stations * N_trials;

    Kokkos::parallel_for(
        "polarity_misfit",
        Kokkos::RangePolicy<ExecSpace>(0, N_total),
        KOKKOS_LAMBDA(const int idx) {
            const int station = idx / N_trials;
            const int trial   = idx % N_trials;

            double obs = obs_pol(station);

            // Check for missing polarity — NaN sentinel
            if (Kokkos::isnan(obs)) {
                misfit(station, trial) = Kokkos::ArithTraits<double>::nan();
                return;
            }

            // Compute dot product: Σᵢ pol_vec[station][i] * mt[trial][i]
            double dot = 0.0;
            for (int c = 0; c < 6; ++c) {
                dot += pol_vec(station, c) * mt(trial, c);
            }

            // Determine synthetic polarity sign
            int syn_pol = (dot > 0.0) ? 1 : ((dot < 0.0) ? -1 : 0);

            // Zero obs_pol with zero pol_vec → skip (not applicable)
            if (obs == 0.0 && syn_pol == 0) {
                misfit(station, trial) = Kokkos::ArithTraits<double>::nan();
                return;
            }

            int obs_int = (obs > 0.5) ? 1 : ((obs < -0.5) ? -1 : 0);

            // Match: sign matches
            misfit(station, trial) = (syn_pol == obs_int) ? 0.0 : 1.0;
        });
}

} // namespace fm

#endif // POLARITY_KERNEL_H