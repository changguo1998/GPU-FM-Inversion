#ifndef XCORR_KERNEL_H
#define XCORR_KERNEL_H

/// XCorr GPU misfit kernel — time-domain cross-correlation misfit.
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
/// Verification invariant: mᵀ·synamp·m = ‖GF·m‖²  (Gram matrix identity)

#include <Kokkos_Core.hpp>
#include <cmath>

namespace fm {

/// Launch the XCorr misfit kernel over (N_phases × N_trials) flat work items.
///
/// @tparam ExecSpace  Kokkos execution space (e.g. Serial, Cuda)
///
/// @param mt            Moment tensor data [6 × N_trials], column-major (LayoutLeft)
/// @param cc_data       Precomputed CC(obs, GF[:,i])    [N_phases·cc_pp × 6]
///                      Phase p rows: [p·cc_pp, (p+1)·cc_pp), each row = one lag
/// @param synamp_data   GF auto-correlation per phase   [N_phases × 36]
///                      Row p: 6×6 sym matrix flattened (i,j) → i·6+j
/// @param obs_norm2     ‖obs‖² per phase                [N_phases]
/// @param[out] misfit   Output misfit per (phase,trial) [N_phases × N_trials]
/// @param N_phases      Number of phases (phase-station pairs)
/// @param N_trials      Number of trials
/// @param cc_pp         CC lags per phase (= 2·maxlag + 1)
template <typename ExecSpace>
inline void launch_xcorr_misfit(
    ExecSpace exec,
    Kokkos::View<const double**, Kokkos::LayoutLeft> mt,
    Kokkos::View<const double**, Kokkos::LayoutLeft> cc_data,
    Kokkos::View<const double**, Kokkos::LayoutLeft> synamp_data,
    Kokkos::View<const double*> obs_norm2,
    Kokkos::View<double**, Kokkos::LayoutLeft> misfit,
    int N_phases,
    int N_trials,
    int cc_pp)
{
    const int total_work = N_phases * N_trials;

    Kokkos::parallel_for(
        "xcorr_misfit",
        Kokkos::RangePolicy<ExecSpace>(exec, 0, total_work),
        KOKKOS_LAMBDA(const int idx) {
            const int phase = idx / N_trials;
            const int trial = idx % N_trials;

            // ── Load moment tensor (6-comp) for this trial (column-major) ──
            double m[6];
            for (int c = 0; c < 6; ++c) {
                m[c] = mt(c, trial);
            }

            // ── syn_norm² = mᵀ · synamp · m  (6×6 quadratic form) ──
            double syn_norm2 = 0.0;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    syn_norm2 += m[i] * synamp_data(phase, i * 6 + j) * m[j];
                }
            }

            const double obs_n2 = obs_norm2(phase);

            // Guard: zero or negative norms → worst misfit (no information)
            if (syn_norm2 <= 0.0 || obs_n2 <= 0.0) {
                misfit(phase, trial) = 1.0;
                return;
            }

            const double denom = sqrt(obs_n2 * syn_norm2);

            // ── cc_syn[k] = Σᵢ m[i] · CC[phase][k][i], then max-normalize ──
            const int cc_start = phase * cc_pp;
            double max_abs_cc = 0.0;

            for (int k = 0; k < cc_pp; ++k) {
                double cc_syn = 0.0;
                for (int i = 0; i < 6; ++i) {
                    cc_syn += m[i] * cc_data(cc_start + k, i);
                }
                double cc_norm = cc_syn / denom;
                double abs_cc = (cc_norm < 0.0) ? -cc_norm : cc_norm;  // fabs
                if (abs_cc > max_abs_cc) {
                    max_abs_cc = abs_cc;
                }
            }

            misfit(phase, trial) = 1.0 - max_abs_cc;
        });
}

} // namespace fm

#endif // XCORR_KERNEL_H