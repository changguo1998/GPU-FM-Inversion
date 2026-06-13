#ifndef PSR_KERNEL_H
#define PSR_KERNEL_H

#include <Kokkos_Core.hpp>

// ─────────────────────────────────────────────────────────
// PSR (P-to-S amplitude ratio) misfit kernel
//
// Misfit formula:
//   syn_amp_P = √(mᵀ · amp_P · m)
//   syn_amp_S = √(mᵀ · amp_S · m)
//   misfit    = (log₁₀(syn_amp_P / syn_amp_S) - obs_psr)²
//
// Inputs:
//   mt       [N_trials × 6]        — moment tensor components per trial
//   amp_P    [N_stations × 6 × 6]  — P-wave amplitude covariance matrix per station
//   amp_S    [N_stations × 6 × 6]  — S-wave amplitude covariance matrix per station
//   obs_psr  [N_stations]          — observed log10(P/S) ratio
//
// Output:
//   misfit   [N_stations × N_trials] — squared log-ratio difference
//
// Missing stations (obs_psr is NaN, or both amp matrices near-zero):
// returns NaN.
//
// Launch: Kokkos::RangePolicy over [0, N_stations * N_trials)
//         flat index = station * N_trials + trial
// ─────────────────────────────────────────────────────────

namespace fm {

template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
void launch_psr_kernel(
    const Kokkos::View<const double**, Kokkos::LayoutLeft, ExecSpace>& mt,         // N_trials × 6
    const Kokkos::View<const double***, Kokkos::LayoutLeft, ExecSpace>& amp_P,     // N_stations × 6 × 6
    const Kokkos::View<const double***, Kokkos::LayoutLeft, ExecSpace>& amp_S,     // N_stations × 6 × 6
    const Kokkos::View<const double*,  ExecSpace>& obs_psr,                        // N_stations
    Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>& misfit)                 // N_stations × N_trials
{
    const int N_stations = amp_P.extent(0);
    const int N_trials   = mt.extent(0);
    const int N_total = N_stations * N_trials;

    Kokkos::parallel_for(
        "psr_misfit",
        Kokkos::RangePolicy<ExecSpace>(0, N_total),
        KOKKOS_LAMBDA(const int idx) {
            const int station = idx / N_trials;
            const int trial   = idx % N_trials;

            double obs = obs_psr(station);
            if (Kokkos::isnan(obs)) {
                misfit(station, trial) = Kokkos::ArithTraits<double>::nan();
                return;
            }

            // Compute quadratic forms:
            // amp_P_quad = Σᵢⱼ amp_P[station][i][j] * mt[trial][i] * mt[trial][j]
            double amp_P_quad = 0.0;
            double amp_S_quad = 0.0;
            for (int i = 0; i < 6; ++i) {
                double m_i = mt(trial, i);
                for (int j = 0; j < 6; ++j) {
                    double m_j = mt(trial, j);
                    amp_P_quad += amp_P(station, i, j) * m_i * m_j;
                    amp_S_quad += amp_S(station, i, j) * m_i * m_j;
                }
            }

            // Numerical safety: clamp small negative values to zero
            if (amp_P_quad < 0.0) amp_P_quad = 0.0;
            if (amp_S_quad < 0.0) amp_S_quad = 0.0;

            double syn_amp_P = Kokkos::sqrt(amp_P_quad);
            double syn_amp_S = Kokkos::sqrt(amp_S_quad);

            // Skip if amplitude ratio is near-zero (degenerate)
            if (syn_amp_P < 1e-30 || syn_amp_S < 1e-30) {
                misfit(station, trial) = Kokkos::ArithTraits<double>::nan();
                return;
            }

            double syn_psr = Kokkos::log10(syn_amp_P / syn_amp_S);
            double diff = syn_psr - obs;

            misfit(station, trial) = diff * diff;
        });
}

} // namespace fm

#endif // PSR_KERNEL_H