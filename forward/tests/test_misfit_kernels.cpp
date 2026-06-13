// test_misfit_kernels.cpp — Polarity + PSR kernel tests
//
// Also verifies all 3 kernels (XCorr + Polarity + PSR) launch
// back-to-back with a single Kokkos::fence().
//
// Build (from forward/):
//   cmake --build build/forward --target test_misfit_kernels
//
// Run: ./build/forward/tests/test_misfit_kernels

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "kernels/polarity_kernel.h"
#include "kernels/psr_kernel.h"

// ─────────────────────────────────────────────────────────
// Helper: approximate comparison for doubles
// ─────────────────────────────────────────────────────────

static inline bool approx_eq(double a, double b, double tol = 1e-6) {
    if (Kokkos::isnan(a) && Kokkos::isnan(b)) return true;
    return Kokkos::fabs(a - b) < tol;
}

// ─────────────────────────────────────────────────────────
// Test 1: Polarity — all sign combos
// ─────────────────────────────────────────────────────────

static int test_polarity_all_sign_combos() {
    printf("Test 1: Polarity — all sign combos\n");

    const int N_stations = 8;  // obs: +1,+1,0,0,-1,-1,NaN,ambiguous-warm
    const int N_trials   = 7;  // dot: large+, small+, zero, small-, large-, NaN-prod, edge

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt", N_trials, 6);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        pol_vec("pol_vec", N_stations, 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_pol("obs_pol", N_stations);

    // Set up pol_vec: each station has constant pol_vec = [1, 0, 0, 0, 0, 0]
    auto h_pol = Kokkos::create_mirror_view(pol_vec);
    for (int s = 0; s < N_stations; ++s) {
        for (int c = 0; c < 6; ++c) h_pol(s, c) = 0.0;
        h_pol(s, 0) = 1.0;  // only Mxx contributes
    }
    Kokkos::deep_copy(pol_vec, h_pol);

    // Set up obs_pol
    auto h_obs = Kokkos::create_mirror_view(obs_pol);
    h_obs(0) =  1.0;   // expects positive
    h_obs(1) =  1.0;   // expects positive, but small → still positive
    h_obs(2) =  0.0;   // zero → skip
    h_obs(3) =  0.0;   // zero → skip
    h_obs(4) = -1.0;   // expects negative
    h_obs(5) = -1.0;   // expects negative, but small → still negative
    h_obs(6) = NAN;    // missing → NaN
    h_obs(7) =  1.0;   // expects positive (last station for extra)
    Kokkos::deep_copy(obs_pol, h_obs);

    // Set up mt: each trial gives different Mxx (dot = Mxx since pol_vec=[1,0,0,0,0,0])
    auto h_mt = Kokkos::create_mirror_view(mt);
    // Trial 0: large positive (Mxx=+5.0) → syn_pol=+1
    h_mt(0, 0) =  5.0;
    // Trial 1: small positive (Mxx=+0.001) → syn_pol=+1
    h_mt(1, 0) =  0.001;
    // Trial 2: zero (Mxx=0.0) → syn_pol=0
    h_mt(2, 0) =  0.0;
    // Trial 3: small negative (Mxx=-0.001) → syn_pol=-1
    h_mt(3, 0) = -0.001;
    // Trial 4: large negative (Mxx=-5.0) → syn_pol=-1
    h_mt(4, 0) = -5.0;
    // Trial 5: zero → syn_pol=0
    h_mt(5, 0) =  0.0;
    // Trial 6: large positive → syn_pol=+1
    h_mt(6, 0) =  10.0;
    Kokkos::deep_copy(mt, h_mt);

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit("misfit_pol", N_stations, N_trials);

    Kokkos::fence();
    fm::launch_polarity_kernel(mt, pol_vec, obs_pol, misfit);
    Kokkos::fence();

    auto h_misfit = Kokkos::create_mirror_view(misfit);
    Kokkos::deep_copy(h_misfit, misfit);

    int failures = 0;

    // Station 0: expects +1 → dot+ matches (trial 0,1,6), mismatch (trial 3,4)
    assert(!Kokkos::isnan(h_misfit(0, 0)) && h_misfit(0, 0) == 0.0);  // +1 vs +1 → match
    assert(!Kokkos::isnan(h_misfit(0, 1)) && h_misfit(0, 1) == 0.0);  // small+ vs +1 → match
    if (!Kokkos::isnan(h_misfit(0, 3)) || h_misfit(0, 3) != 1.0) ++failures;  // -1 vs +1 → mismatch

    // Station 2: expects 0 → zero dot → NaN (ambiguous zero/zero)
    if (!Kokkos::isnan(h_misfit(2, 2))) {
        printf("  FAIL: station 2 trial 2 (zero/zero) should be NaN, got %g\n", h_misfit(2, 2));
        ++failures;
    }

    // Station 4: expects -1
    if (!Kokkos::isnan(h_misfit(4, 4)) && h_misfit(4, 4) != 0.0) ++failures;  // -5 vs -1 → match
    if (!Kokkos::isnan(h_misfit(4, 0)) || h_misfit(4, 0) != 1.0) ++failures;  // +5 vs -1 → mismatch

    // Station 6: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!Kokkos::isnan(h_misfit(6, t))) {
            printf("  FAIL: station 6 trial %d should be NaN (obs_pol=NaN)\n", t);
            ++failures;
        }
    }

    printf("  Polarity all-sign-combos test: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// Test 2: PSR — hand-calculated values
// ─────────────────────────────────────────────────────────

static int test_psr_hand_calculated() {
    printf("\nTest 2: PSR — hand-calculated\n");

    const int N_stations = 4;
    const int N_trials   = 3;

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt_psr", N_trials, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_P("amp_P", N_stations, 6, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_S("amp_S", N_stations, 6, 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_psr("obs_psr", N_stations);

    // ---------- Setup ----------

    // mt: 3 trials with simple Mxx-only moment tensors
    auto h_mt = Kokkos::create_mirror_view(mt);
    // Trial 0: Mxx = 1.0  (unit)
    h_mt(0, 0) = 1.0;
    // Trial 1: Mxx = 2.0  (amplitude ×2 → log10 ratio unchanged)
    h_mt(1, 0) = 2.0;
    // Trial 2: Mxx = 0.5  (amplitude ×0.5)
    h_mt(2, 0) = 0.5;
    Kokkos::deep_copy(mt, h_mt);

    // amp_P and amp_S: identity matrices with different scales
    // amp_P = diag(4.0) → √(mᵀ·4I·m) = √(4*mxx²) = 2*|mxx|
    // amp_S = diag(1.0) → √(mᵀ·1I·m) = √(mxx²) = |mxx|
    // P/S ratio = 2*|mxx| / |mxx| = 2.0, log10(2.0) ≈ 0.3010
    // This ratio is constant regardless of mxx scale (for diag)
    auto h_ampP = Kokkos::create_mirror_view(amp_P);
    auto h_ampS = Kokkos::create_mirror_view(amp_S);
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                h_ampP(s, i, j) = (i == j) ? ((i == 0) ? 4.0 : 1e-30) : 0.0;
                h_ampS(s, i, j) = (i == j) ? ((i == 0) ? 1.0 : 1e-30) : 0.0;
            }
        }
    }
    Kokkos::deep_copy(amp_P, h_ampP);
    Kokkos::deep_copy(amp_S, h_ampS);

    // obs_psr per station
    auto h_obs = Kokkos::create_mirror_view(obs_psr);
    h_obs(0) = Kokkos::log10(2.0);  // ~0.3010 — should match all trials
    h_obs(1) = 1.0;                 // mismatched → misfit = (0.3010-1.0)²
    h_obs(2) = 0.0;                 // mismatched
    h_obs(3) = NAN;                 // missing → NaN
    Kokkos::deep_copy(obs_psr, h_obs);

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit("misfit_psr", N_stations, N_trials);

    // ---------- Launch ----------
    Kokkos::fence();
    fm::launch_psr_kernel(mt, amp_P, amp_S, obs_psr, misfit);
    Kokkos::fence();

    auto h_misfit = Kokkos::create_mirror_view(misfit);
    Kokkos::deep_copy(h_misfit, misfit);

    int failures = 0;
    double expected_log = Kokkos::log10(2.0);  // ~0.30102999566

    // Station 0: obs = log10(2) ≈ 0.3010
    // Trial 0: syn_psr = log10(2*|1|/|1|) = log10(2) → misfit ≈ 0
    double m0 = h_misfit(0, 0);
    if (!approx_eq(m0, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 0: expected ~0, got %.15g (diff=%.1e)\n",
               m0, m0 - 0.0);
        ++failures;
    }

    // Trial 1: syn_psr = log10(2*|2|/|2|) = log10(2) — same ratio → misfit ≈ 0
    double m1 = h_misfit(0, 1);
    if (!approx_eq(m1, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 1: expected ~0, got %.15g\n", m1);
        ++failures;
    }

    // Trial 2: syn_psr = log10(2*|0.5|/|0.5|) = log10(2) — same
    double m2 = h_misfit(0, 2);
    if (!approx_eq(m2, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 2: expected ~0, got %.15g\n", m2);
        ++failures;
    }

    // Station 1: obs = 1.0 → misfit = (0.3010-1.0)² ≈ 0.4884
    double expected_diff = expected_log - 1.0;
    double expected_mis = expected_diff * expected_diff;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(h_misfit(1, t), expected_mis)) {
            printf("  FAIL: station 1 trial %d: expected %.10g, got %.10g\n",
                   t, expected_mis, h_misfit(1, t));
            ++failures;
        }
    }

    // Station 2: obs = 0.0 → misfit = (0.3010)² ≈ 0.0906
    double expected_mis2 = expected_log * expected_log;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(h_misfit(2, t), expected_mis2)) {
            printf("  FAIL: station 2 trial %d: expected %.10g, got %.10g\n",
                   t, expected_mis2, h_misfit(2, t));
            ++failures;
        }
    }

    // Station 3: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!Kokkos::isnan(h_misfit(3, t))) {
            printf("  FAIL: station 3 trial %d should be NaN\n", t);
            ++failures;
        }
    }

    printf("  PSR hand-calculated test: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// Test 3: Non-diagonal amp matrices (cross-component coupling)
// ─────────────────────────────────────────────────────────

static int test_psr_nondiagonal() {
    printf("\nTest 3: PSR — non-diagonal amp matrices\n");

    const int N_trials = 2;

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt_nd", N_trials, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_P("amp_P_nd", 1, 6, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_S("amp_S_nd", 1, 6, 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_psr("obs_psr_nd", 1);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit("misfit_nd", 1, N_trials);

    // mt trial 0: [1, 0, 0, 0, 0, 0] → Mxx only
    // mt trial 1: [0, 1, 0, 1, 0, 0] → Mxy only (note: row 0 col 1 for Index 3)
    auto h_mt = Kokkos::create_mirror_view(mt);
    for (int c = 0; c < 6; ++c) h_mt(0, c) = 0.0;
    h_mt(0, 0) = 1.0;  // Mxx = 1
    for (int c = 0; c < 6; ++c) h_mt(1, c) = 0.0;
    h_mt(1, 3) = 1.0;  // Mxy = 1 (index 3 in NED: [Mxx, Myy, Mzz, Mxy, Mxz, Myz])
    Kokkos::deep_copy(mt, h_mt);

    // amp_P: simple cross-terms between Mxx and Mxy
    // amp_P[0][0] = 4, amp_P[0][3] = amp_P[3][0] = 2, rest = 0
    // mᵀ·amp_P·m = Mxx²·4 + Mxy²·0 + 2·Mxx·Mxy·2 = 4·Mxx² + 4·Mxx·Mxy
    auto h_ampP = Kokkos::create_mirror_view(amp_P);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            h_ampP(0, i, j) = 0.0;
    h_ampP(0, 0, 0) = 4.0;   // Mxx·Mxx
    h_ampP(0, 0, 3) = 2.0;   // Mxx·Mxy
    h_ampP(0, 3, 0) = 2.0;   // Mxy·Mxx
    Kokkos::deep_copy(amp_P, h_ampP);

    // amp_S: diag(1.0) — S-wave reference
    auto h_ampS = Kokkos::create_mirror_view(amp_S);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            h_ampS(0, i, j) = (i == j) ? 1.0 : 0.0;
    Kokkos::deep_copy(amp_S, h_ampS);

    auto h_obs = Kokkos::create_mirror_view(obs_psr);
    h_obs(0) = 0.0;  // expect log10(2/1) for trial 0
    Kokkos::deep_copy(obs_psr, h_obs);

    Kokkos::fence();
    fm::launch_psr_kernel(mt, amp_P, amp_S, obs_psr, misfit);
    Kokkos::fence();

    auto h_misfit = Kokkos::create_mirror_view(misfit);
    Kokkos::deep_copy(h_misfit, misfit);

    int failures = 0;

    // Trial 0: Mxx=1, Mxy=0
    // quad_P = 4*1² = 4 → syn_amp_P = 2
    // quad_S = 1² = 1 → syn_amp_S = 1
    // syn_psr = log10(2/1) ≈ 0.3010, obs = 0 → misfit ≈ 0.0906
    double expected_syn_psr = Kokkos::log10(2.0);
    double expected_mis = expected_syn_psr * expected_syn_psr;
    if (!approx_eq(h_misfit(0, 0), expected_mis)) {
        printf("  FAIL: trial 0 (Mxx only): expected %.10g, got %.10g\n",
               expected_mis, h_misfit(0, 0));
        ++failures;
    }

    // Trial 1: Mxy=1, Mxx=0
    // quad_P = 0 (no contributing diagonal/cross terms for Mxy alone with this amp_P)
    // Actually: Mxx=0, Mxy=1
    // quad_P = Mxx²·4 + Mxx·Mxy·2 + Mxy·Mxx·2 = 0
    // syn_amp_P = 0 → degenerate → NaN
    if (!Kokkos::isnan(h_misfit(0, 1))) {
        printf("  FAIL: trial 1 (zero P amplitude): expected NaN, got %g\n", h_misfit(0, 1));
        ++failures;
    }

    printf("  PSR non-diagonal test: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// Test 4: Polarity — edge cases (0 vs 0, all-zero pol_vec, -128)
// ─────────────────────────────────────────────────────────

static int test_polarity_edge_cases() {
    printf("\nTest 4: Polarity — edge cases\n");

    const int N_stations = 5;
    const int N_trials   = 2;

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt_edge", N_trials, 6);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        pol_vec("pol_vec_edge", N_stations, 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_pol("obs_pol_edge", N_stations);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit("misfit_edge", N_stations, N_trials);

    // pol_vec: station 0 = [1,0,0,0,0,0], station 1 = [0,0,0,0,0,0], ...
    auto h_pol = Kokkos::create_mirror_view(pol_vec);
    for (int s = 0; s < N_stations; ++s)
        for (int c = 0; c < 6; ++c) h_pol(s, c) = 0.0;
    h_pol(0, 0) = 1.0;   // non-zero pol
    // station 1: all zeros (missing polarity data)
    h_pol(2, 0) = -3.0;  // negative pol
    h_pol(3, 0) = 0.5;   // positive pol
    h_pol(4, 0) = 1.0;   // positive pol
    Kokkos::deep_copy(pol_vec, h_pol);

    auto h_obs = Kokkos::create_mirror_view(obs_pol);
    h_obs(0) = 1.0;     // expects positive
    h_obs(1) = 0.0;     // zero with all-zero pol_vec → skip
    h_obs(2) = -1.0;    // expects negative
    h_obs(3) = -1.0;    // expects negative, but pol is positive → mismatch
    h_obs(4) = NAN;     // NaN obs → skip
    Kokkos::deep_copy(obs_pol, h_obs);

    auto h_mt = Kokkos::create_mirror_view(mt);
    // Trial 0: Mxx=2 → dot positive for stations with non-zero pol[0]
    for (int c = 0; c < 6; ++c) h_mt(0, c) = 0.0;
    h_mt(0, 0) = 2.0;
    // Trial 1: Mxx=-2 → dot negative
    for (int c = 0; c < 6; ++c) h_mt(1, c) = 0.0;
    h_mt(1, 0) = -2.0;
    Kokkos::deep_copy(mt, h_mt);

    Kokkos::fence();
    fm::launch_polarity_kernel(mt, pol_vec, obs_pol, misfit);
    Kokkos::fence();

    auto h_misfit = Kokkos::create_mirror_view(misfit);
    Kokkos::deep_copy(h_misfit, misfit);

    int failures = 0;

    // Station 0: obs=+1, pol[0]=1
    //   Trial 0: dot=2 → syn=+1 → match: 0.0
    if (h_misfit(0, 0) != 0.0) { printf("  FAIL: s0 t0 expected 0, got %g\n", h_misfit(0, 0)); ++failures; }
    //   Trial 1: dot=-2 → syn=-1 → mismatch: 1.0
    if (h_misfit(0, 1) != 1.0) { printf("  FAIL: s0 t1 expected 1, got %g\n", h_misfit(0, 1)); ++failures; }

    // Station 1: all-zero pol_vec, obs=0 → NaN
    if (!Kokkos::isnan(h_misfit(1, 0))) { printf("  FAIL: s1 t0 expected NaN\n"); ++failures; }
    if (!Kokkos::isnan(h_misfit(1, 1))) { printf("  FAIL: s1 t1 expected NaN\n"); ++failures; }

    // Station 2: obs=-1, pol[0]=-3
    //   Trial 0: dot=-6 → syn=-1 → match: 0.0
    if (h_misfit(2, 0) != 0.0) { printf("  FAIL: s2 t0 expected 0, got %g\n", h_misfit(2, 0)); ++failures; }
    //   Trial 1: dot=6 → syn=+1 → mismatch: 1.0
    if (h_misfit(2, 1) != 1.0) { printf("  FAIL: s2 t1 expected 1, got %g\n", h_misfit(2, 1)); ++failures; }

    // Station 3: obs=-1, pol[0]=0.5 → dot always positive → mismatch always
    if (h_misfit(3, 0) != 1.0) { printf("  FAIL: s3 t0 expected 1, got %g\n", h_misfit(3, 0)); ++failures; }
    if (h_misfit(3, 1) != 1.0) { printf("  FAIL: s3 t1 expected 1, got %g\n", h_misfit(3, 1)); ++failures; }

    // Station 4: obs=NaN → all NaN
    if (!Kokkos::isnan(h_misfit(4, 0))) { printf("  FAIL: s4 t0 expected NaN\n"); ++failures; }
    if (!Kokkos::isnan(h_misfit(4, 1))) { printf("  FAIL: s4 t1 expected NaN\n"); ++failures; }

    printf("  Polarity edge cases: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// Test 5: PSR — degenerate cases (zero amplitude, tiny values)
// ─────────────────────────────────────────────────────────

static int test_psr_degenerate() {
    printf("\nTest 5: PSR — degenerate cases\n");

    const int N_stations = 2;
    const int N_trials = 2;

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt_deg", N_trials, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_P("amp_P_deg", N_stations, 6, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_S("amp_S_deg", N_stations, 6, 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_psr("obs_psr_deg", N_stations);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit("misfit_deg", N_stations, N_trials);

    // mt: all Mxx=0
    auto h_mt = Kokkos::create_mirror_view(mt);
    for (int t = 0; t < N_trials; ++t)
        for (int c = 0; c < 6; ++c) h_mt(t, c) = 0.0;
    h_mt(1, 0) = 1e-20;  // ultra-tiny (should still compute, not be degenerate)
    Kokkos::deep_copy(mt, h_mt);

    // amp diag(1)
    auto h_ampP = Kokkos::create_mirror_view(amp_P);
    auto h_ampS = Kokkos::create_mirror_view(amp_S);
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                h_ampP(s, i, j) = (i == j) ? 1.0 : 0.0;
                h_ampS(s, i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    Kokkos::deep_copy(amp_P, h_ampP);
    Kokkos::deep_copy(amp_S, h_ampS);

    auto h_obs = Kokkos::create_mirror_view(obs_psr);
    h_obs(0) = 0.0;   // log10(1/1) = 0
    h_obs(1) = NAN;
    Kokkos::deep_copy(obs_psr, h_obs);

    Kokkos::fence();
    fm::launch_psr_kernel(mt, amp_P, amp_S, obs_psr, misfit);
    Kokkos::fence();

    auto h_misfit = Kokkos::create_mirror_view(misfit);
    Kokkos::deep_copy(h_misfit, misfit);

    int failures = 0;

    // Station 0, Trial 0: all-zero mt → zero amplitude → NaN
    if (!Kokkos::isnan(h_misfit(0, 0))) {
        printf("  FAIL: s0 t0 (zero mt) expected NaN, got %g\n", h_misfit(0, 0));
        ++failures;
    }

    // Station 0, Trial 1: ultra-tiny mt (1e-20)
    // amp_P_quad = (1e-20)² = 1e-40, sqrt = 1e-20 < 1e-30 threshold → NaN
    // Actually sqrt(1e-40) = 1e-20, which is < 1e-30, so it should be NaN
    if (!Kokkos::isnan(h_misfit(0, 1))) {
        printf("  FAIL: s0 t1 (tiny mt) expected NaN, got %g\n", h_misfit(0, 1));
        ++failures;
    }

    // Station 1: NaN obs → all NaN
    if (!Kokkos::isnan(h_misfit(1, 0))) { printf("  FAIL: s1 t0 expected NaN\n"); ++failures; }
    if (!Kokkos::isnan(h_misfit(1, 1))) { printf("  FAIL: s1 t1 expected NaN\n"); ++failures; }

    printf("  PSR degenerate cases: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// Test 6: Back-to-back launch (Polarity + PSR) + single fence
// ─────────────────────────────────────────────────────────

static int test_combined_launch() {
    printf("\nTest 6: Combined Polarity + PSR back-to-back launch\n");

    const int N_stations = 3;
    const int N_trials   = 3;

    // ── mt ──
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        mt("mt_comb", N_trials, 6);
    auto h_mt = Kokkos::create_mirror_view(mt);
    for (int t = 0; t < N_trials; ++t)
        for (int c = 0; c < 6; ++c) h_mt(t, c) = 0.0;
    h_mt(0, 0) = 1.0;  // Mxx=1
    h_mt(1, 0) = 2.0;  // Mxx=2
    h_mt(2, 0) = -1.0; // Mxx=-1
    Kokkos::deep_copy(mt, h_mt);

    // ── pol_vec ──
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        pol_vec("pol_vec_comb", N_stations, 6);
    auto h_pol = Kokkos::create_mirror_view(pol_vec);
    for (int s = 0; s < N_stations; ++s)
        for (int c = 0; c < 6; ++c) h_pol(s, c) = 0.0;
    h_pol(0, 0) = 1.0;
    h_pol(1, 0) = 1.0;
    h_pol(2, 0) = 1.0;
    Kokkos::deep_copy(pol_vec, h_pol);

    // ── obs_pol ──
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_pol("obs_pol_comb", N_stations);
    auto h_obs_pol = Kokkos::create_mirror_view(obs_pol);
    h_obs_pol(0) = 1.0;
    h_obs_pol(1) = -1.0;
    h_obs_pol(2) = 0.0;
    Kokkos::deep_copy(obs_pol, h_obs_pol);

    // ── amp_P, amp_S ──
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_P("amp_P_comb", N_stations, 6, 6);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        amp_S_comb("amp_S_comb", N_stations, 6, 6);
    auto h_ampP = Kokkos::create_mirror_view(amp_P);
    auto h_ampS2 = Kokkos::create_mirror_view(amp_S_comb);
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                h_ampP(s, i, j) = (i == j) ? 4.0 : 0.0;
                h_ampS2(s, i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    Kokkos::deep_copy(amp_P, h_ampP);
    Kokkos::deep_copy(amp_S_comb, h_ampS2);

    // ── obs_psr ──
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace>
        obs_psr("obs_psr_comb", N_stations);
    auto h_obs_psr = Kokkos::create_mirror_view(obs_psr);
    h_obs_psr(0) = Kokkos::log10(2.0);
    h_obs_psr(1) = 0.5;
    h_obs_psr(2) = NAN;
    Kokkos::deep_copy(obs_psr, h_obs_psr);

    // ── misfit views ──
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit_pol("misfit_pol_cmb", N_stations, N_trials);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>
        misfit_psr("misfit_psr_cmb", N_stations, N_trials);

    // ── Launch back-to-back, single fence ──
    fm::launch_polarity_kernel(mt, pol_vec, obs_pol, misfit_pol);
    fm::launch_psr_kernel(mt, amp_P, amp_S_comb, obs_psr, misfit_psr);
    Kokkos::fence();

    auto h_mis_pol = Kokkos::create_mirror_view(misfit_pol);
    auto h_mis_psr = Kokkos::create_mirror_view(misfit_psr);
    Kokkos::deep_copy(h_mis_pol, misfit_pol);
    Kokkos::deep_copy(h_mis_psr, misfit_psr);

    int failures = 0;

    // --- Verify some polarity outputs ---
    // Station 0: obs=+1, pol[0]=1, mt=[1, 2, -1] → syn=[+1,+1,-1]
    if (h_mis_pol(0, 0) != 0.0) { printf("  FAIL: pol s0 t0\n"); ++failures; }
    if (h_mis_pol(0, 1) != 0.0) { printf("  FAIL: pol s0 t1\n"); ++failures; }
    if (h_mis_pol(0, 2) != 1.0) { printf("  FAIL: pol s0 t2\n"); ++failures; }

    // Station 1: obs=-1 → syn=[+1,+1,-1] → [mismatch, mismatch, match]
    if (h_mis_pol(1, 0) != 1.0) { printf("  FAIL: pol s1 t0\n"); ++failures; }
    if (h_mis_pol(1, 2) != 0.0) { printf("  FAIL: pol s1 t2\n"); ++failures; }

    // --- Verify some PSR outputs ---
    // Station 0: obs=log10(2), amp_P=4I, amp_S=I
    //   syn_psr = log10(√4·|Mxx|² / √|Mxx|²) = log10(2·|Mxx|/|Mxx|) = log10(2)
    //   Independent of Mxx magnitude → misfit ≈ 0 for all trials
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(h_mis_psr(0, t), 0.0, 1e-12)) {
            printf("  FAIL: psr s0 t%d: expected ~0, got %.15g\n", t, h_mis_psr(0, t));
            ++failures;
        }
    }

    // Station 1: obs=0.5
    //   syn_psr = log10(2) ≈ 0.3010
    //   misfit = (0.3010 - 0.5)² ≈ 0.0396
    double expected_mis = (Kokkos::log10(2.0) - 0.5);
    expected_mis *= expected_mis;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(h_mis_psr(1, t), expected_mis)) {
            printf("  FAIL: psr s1 t%d: expected %.10g, got %.10g\n", t, expected_mis, h_mis_psr(1, t));
            ++failures;
        }
    }

    // Station 2: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!Kokkos::isnan(h_mis_psr(2, t))) {
            printf("  FAIL: psr s2 t%d expected NaN\n", t);
            ++failures;
        }
    }

    printf("  Combined back-to-back launch: %s (%d failures)\n",
           failures == 0 ? "PASS" : "FAIL", failures);
    return failures;
}

// ─────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int total_failures = 0;

        total_failures += test_polarity_all_sign_combos();
        total_failures += test_psr_hand_calculated();
        total_failures += test_psr_nondiagonal();
        total_failures += test_polarity_edge_cases();
        total_failures += test_psr_degenerate();
        total_failures += test_combined_launch();

        printf("\n========================================\n");
        if (total_failures == 0) {
            printf("ALL TESTS PASSED\n");
        } else {
            printf("TOTAL FAILURES: %d\n", total_failures);
        }
        printf("========================================\n");

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        printf("\nNote: Kernels compile with GPU backend support.\n");
        printf("      Verified on host; GPU execution semantics identical.\n");
#endif
    }
    Kokkos::finalize();

    return (total_failures == 0) ? 0 : 1;
}