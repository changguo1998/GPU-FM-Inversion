/// Test for XCorr GPU misfit kernel.
///
/// Runs on Kokkos::Serial (CPU) to verify:
///  1. Hand-computed reference match for small synthetic data
///  2. Gram matrix identity: mᵀ·synamp·m = ‖GF·m‖²
///  3. Boundary: zero-norm, maxlag=0, single phase, single trial

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "../src/kernels/xcorr_kernel.h"

// helpers
static inline double sqr(double x) { return x * x; }
static inline bool   approx(double a, double b, double tol = 1e-10) {
    double d = (a > b) ? (a - b) : (b - a);
    return d < tol;
}

// ────────────────────────────────────────────────────────────────
// Compute misfit manually (pure C++, no Kokkos) for one (p, t)
// ────────────────────────────────────────────────────────────────
double ref_misfit(int phase, int trial,
                  const double* mt,        // [6 * N_trials] col-major
                  const double* cc_data,    // [N_phases * cc_pp * 6]
                  const double* synamp_data,// [N_phases * 36]
                  const double* obs_norm2,
                  int N_trials, int cc_pp)  // N_trials passed but unused here
{
    const double* m = mt + 6 * trial;

    // syn_norm2
    double syn_n2 = 0.0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            syn_n2 += m[i] * synamp_data[phase * 36 + i * 6 + j] * m[j];

    if (syn_n2 <= 0.0 || obs_norm2[phase] <= 0.0) return 1.0;

    double denom = std::sqrt(obs_norm2[phase] * syn_n2);

    double best = 0.0;
    for (int k = 0; k < cc_pp; ++k) {
        double cc = 0.0;
        for (int i = 0; i < 6; ++i)
            cc += m[i] * cc_data[(phase * cc_pp + k) * 6 + i];
        double v = std::fabs(cc / denom);
        if (v > best) best = v;
    }
    return 1.0 - best;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int passed = 0, total = 0;
        printf("=== XCorr Kernel Tests (Kokkos::Serial) ===\n\n");

        // ── Test 1: Hand-computed reference (2 phases, 3 trials, maxlag=1) ──
        {
            const int Nph = 2, Ntr = 3, cc_pp = 3; // maxlag=1 → 2*1+1 = 3

            // Host arrays
            auto h_mt    = Kokkos::View<double**, Kokkos::LayoutLeft>("h_mt",    6, Ntr);
            auto h_cc    = Kokkos::View<double**, Kokkos::LayoutLeft>("h_cc",    Nph * cc_pp, 6);
            auto h_syn   = Kokkos::View<double**, Kokkos::LayoutLeft>("h_syn",   Nph, 36);
            auto h_obs   = Kokkos::View<double*>("h_obs",                        Nph);
            auto h_mis   = Kokkos::View<double**, Kokkos::LayoutLeft>("h_mis",   Nph, Ntr);

            // Seed
            srand(42);

            // Fill CC data
            for (int p = 0; p < Nph; ++p)
                for (int k = 0; k < cc_pp; ++k)
                    for (int i = 0; i < 6; ++i)
                        h_cc(p * cc_pp + k, i) = (double)(rand() % 1000) / 1000.0 - 0.2;

            // Fill synamp (symmetric positive-definite-ish: identity + noise)
            for (int p = 0; p < Nph; ++p)
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        h_syn(p, i * 6 + j) = (i == j) ? (1.0 + (rand() % 100) / 500.0)
                                                        : ((rand() % 100) / 1000.0 - 0.05);

            // obs_norm2
            h_obs(0) = 3.7;   h_obs(1) = 2.1;

            // MT (random, then normalize)
            for (int t = 0; t < Ntr; ++t) {
                double norm = 0.0;
                for (int c = 0; c < 6; ++c) {
                    double v = (rand() % 2000) / 1000.0 - 1.0;
                    h_mt(c, t) = v;
                    norm += v * v;
                }
                // Normalize so MT has unit norm (helps numerical stability)
                norm = std::sqrt(norm);
                for (int c = 0; c < 6; ++c) h_mt(c, t) /= norm;
            }

            // Run kernel
            fm::launch_xcorr_misfit(Kokkos::Serial(),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(h_mt),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(h_cc),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(h_syn),
                                     h_obs,
                                     h_mis,
                                     Nph, Ntr, cc_pp);
            Kokkos::fence();

            // Verify against reference
            bool ok = true;
            double max_err = 0.0;
            for (int p = 0; p < Nph; ++p) {
                for (int t = 0; t < Ntr; ++t) {
                    double expected = ref_misfit(p, t,
                                                  h_mt.data(),
                                                  h_cc.data(),
                                                  h_syn.data(),
                                                  h_obs.data(), Ntr, cc_pp);
                    double diff = std::fabs(h_mis(p, t) - expected);
                    if (diff > max_err) max_err = diff;
                    if (diff > 1e-10) { ok = false; }
                }
            }
            total++;
            if (ok) { passed++; printf("  [PASS] Test 1: Reference match  (max diff = %e)\n", max_err); }
            else     printf("  [FAIL] Test 1: Reference match  (max diff = %e)\n", max_err);
        }

        // ── Test 2: Gram matrix identity ──
        {
            const int Nsamp = 20; // waveform length
            // Construct a random GF [Nsamp][6]
            srand(123);
            double gf[20 * 6];
            for (int i = 0; i < Nsamp * 6; ++i)
                gf[i] = (double)(rand() % 2000) / 1000.0 - 1.0;

            // Compute synamp = GFᵀ·GF
            double syn[36] = {0};
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    for (int n = 0; n < Nsamp; ++n)
                        syn[i * 6 + j] += gf[n * 6 + i] * gf[n * 6 + j];

            // Random m
            double m[6] = {0.3, 0.1, -0.5, 0.8, -0.2, 0.4};
            // Normalize
            double mn = 0;
            for (int i = 0; i < 6; ++i) mn += m[i] * m[i];
            mn = std::sqrt(mn);
            for (int i = 0; i < 6; ++i) m[i] /= mn;

            // lhs = mᵀ · synamp · m
            double lhs = 0.0;
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    lhs += m[i] * syn[i * 6 + j] * m[j];

            // rhs = ‖GF·m‖²
            double rhs = 0.0;
            for (int n = 0; n < Nsamp; ++n) {
                double dot = 0.0;
                for (int i = 0; i < 6; ++i)
                    dot += gf[n * 6 + i] * m[i];
                rhs += dot * dot;
            }

            total++;
            if (approx(lhs, rhs, 1e-12)) {
                passed++;
                printf("  [PASS] Test 2: Gram identity   lhs=%e  rhs=%e  diff=%e\n",
                       lhs, rhs, std::fabs(lhs - rhs));
            } else {
                printf("  [FAIL] Test 2: Gram identity   lhs=%e  rhs=%e  diff=%e\n",
                       lhs, rhs, std::fabs(lhs - rhs));
            }
        }

        // ── Test 3: Kernel with Gram-consistent synamp ──
        {
            // Use the same GF and synamp from test 2, run through the kernel with a single trial.
            // This verifies end-to-end: GF → synamp → kernel gives correct misfit for known cc=0 case.
            const int Nph = 1, Ntr = 1, cc_pp = 1;

            auto g_mt    = Kokkos::View<double**, Kokkos::LayoutLeft>("g_mt",  6, Ntr);
            auto g_cc    = Kokkos::View<double**, Kokkos::LayoutLeft>("g_cc",  Nph * cc_pp, 6);
            auto g_syn   = Kokkos::View<double**, Kokkos::LayoutLeft>("g_syn", Nph, 36);
            auto g_obs2  = Kokkos::View<double*>("g_obs2",                     Nph);
            auto g_mis   = Kokkos::View<double**, Kokkos::LayoutLeft>("g_mis", Nph, Ntr);

            // Copy synamp from test 2
            for (int k = 0; k < 36; ++k) g_syn(0, k) = syn[k];
            // obs_norm2 = 1.0
            g_obs2(0) = 1.0;
            // cc_data → all 1.0 (perfect correlation at lag 0)
            for (int i = 0; i < 6; ++i) g_cc(0, i) = 1.0;
            // MT = normalized m from test 2
            for (int i = 0; i < 6; ++i) g_mt(i, 0) = m[i];

            fm::launch_xcorr_misfit(Kokkos::Serial(),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(g_mt),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(g_cc),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(g_syn),
                                     g_obs2, g_mis,
                                     Nph, Ntr, cc_pp);
            Kokkos::fence();

            // Expected: cc_syn = Σ m[i] = sum(m), syn_norm2 = lhs (from test 2)
            // denom = sqrt(1.0 * lhs), cc_norm = sum(m)/sqrt(lhs), misfit = 1 - |cc_norm|
            double expected_sum = 0.0;
            for (int i = 0; i < 6; ++i) expected_sum += m[i];

            double misfit_gold = 1.0 - std::fabs(expected_sum / std::sqrt(lhs));

            total++;
            if (approx(g_mis(0, 0), misfit_gold, 1e-12)) {
                passed++;
                printf("  [PASS] Test 3: End-to-end      misfit=%e  expected=%e\n",
                       g_mis(0, 0), misfit_gold);
            } else {
                printf("  [FAIL] Test 3: End-to-end      misfit=%e  expected=%e\n",
                       g_mis(0, 0), misfit_gold);
            }
        }

        // ── Test 4: Boundary — zero syn_norm2 (should return 1.0) ──
        {
            const int Nph = 1, Ntr = 1, cc_pp = 1;

            auto z_mt   = Kokkos::View<double**, Kokkos::LayoutLeft>("z_mt",  6, Ntr);
            auto z_cc   = Kokkos::View<double**, Kokkos::LayoutLeft>("z_cc",  Nph * cc_pp, 6);
            auto z_syn  = Kokkos::View<double**, Kokkos::LayoutLeft>("z_syn", Nph, 36);
            auto z_obs2 = Kokkos::View<double*>("z_obs2",                     Nph);
            auto z_mis  = Kokkos::View<double**, Kokkos::LayoutLeft>("z_mis", Nph, Ntr);

            for (int k = 0; k < 36; ++k) z_syn(0, k) = 0.0;  // zero synamp
            z_obs2(0)    = 0.0;
            z_cc(0, 0)   = 1.0;
            for (int i = 0; i < 5; ++i) z_cc(0, i + 1) = 0.0;
            for (int i = 0; i < 6; ++i) z_mt(i, 0) = 0.0;

            fm::launch_xcorr_misfit(Kokkos::Serial(),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(z_mt),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(z_cc),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(z_syn),
                                     z_obs2, z_mis,
                                     Nph, Ntr, cc_pp);
            Kokkos::fence();

            total++;
            if (z_mis(0, 0) == 1.0) {
                passed++;
                printf("  [PASS] Test 4: Zero-norm guard    misfit=%.1f\n", z_mis(0, 0));
            } else {
                printf("  [FAIL] Test 4: Zero-norm guard    misfit=%e (expected 1.0)\n",
                       z_mis(0, 0));
            }
        }

        // ── Test 5: Boundary — single phase, single trial, maxlag=0 (cc_pp=1) ──
        {
            const int Nph = 1, Ntr = 1, cc_pp = 1;

            auto s_mt   = Kokkos::View<double**, Kokkos::LayoutLeft>("s_mt",  6, Ntr);
            auto s_cc   = Kokkos::View<double**, Kokkos::LayoutLeft>("s_cc",  Nph * cc_pp, 6);
            auto s_syn  = Kokkos::View<double**, Kokkos::LayoutLeft>("s_syn", Nph, 36);
            auto s_obs2 = Kokkos::View<double*>("s_obs2",                     Nph);
            auto s_mis  = Kokkos::View<double**, Kokkos::LayoutLeft>("s_mis", Nph, Ntr);

            // Simple case: m=I6[0], synamp=I, obs_norm2=1, cc=[1,0,0,0,0,0]
            // Then cc_syn = 1, syn_norm2 = 1, cc_norm=1, misfit=0
            for (int k = 0; k < 6; ++k) s_mt(k, 0) = 0.0;
            s_mt(0, 0) = 1.0;       // m = [1,0,0,0,0,0]

            s_cc(0, 0) = 1.0;       // CC[0] = [1,0,0,0,0,0]
            for (int i = 1; i < 6; ++i) s_cc(0, i) = 0.0;

            for (int k = 0; k < 36; ++k) s_syn(0, k) = 0.0;
            s_syn(0, 0) = 1.0;  // synamp = I6

            s_obs2(0) = 1.0;

            fm::launch_xcorr_misfit(Kokkos::Serial(),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(s_mt),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(s_cc),
                                     Kokkos::View<const double**, Kokkos::LayoutLeft>(s_syn),
                                     s_obs2, s_mis,
                                     Nph, Ntr, cc_pp);
            Kokkos::fence();

            total++;
            if (approx(s_mis(0, 0), 0.0, 1e-12)) {
                passed++;
                printf("  [PASS] Test 5: Minimal 1×1×1       misfit=%e\n", s_mis(0, 0));
            } else {
                printf("  [FAIL] Test 5: Minimal 1×1×1       misfit=%e (expected 0)\n",
                       s_mis(0, 0));
            }
        }

        printf("\n=== %d/%d tests passed ===\n", passed, total);
        if (passed != total) {
            Kokkos::finalize();
            return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}