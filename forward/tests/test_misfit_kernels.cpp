// test_misfit_kernels.cpp — Polarity + PSR kernel tests
//
// Also verifies both kernels launch back-to-back.
//
// Build (from forward/):
//   cmake --build build/forward --target test_misfit_kernels
//
// Run: ./build/forward/tests/test_misfit_kernels

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>

#include "kernels/polarity_kernel.h"
#include "kernels/psr_kernel.h"

// ─────────────────────────────────────────────────────────
// Helper: approximate comparison for doubles
// ─────────────────────────────────────────────────────────

static inline bool approx_eq(double a, double b, double tol = 1e-6) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::fabs(a - b) < tol;
}

// ─────────────────────────────────────────────────────────
// Test 1: Polarity — all sign combos
// ─────────────────────────────────────────────────────────

static int test_polarity_all_sign_combos() {
    printf("Test 1: Polarity — all sign combos\n");

    const int N_stations = 8;  // obs: +1,+1,0,0,-1,-1,NaN,ambiguous-warm
    const int N_trials   = 7;  // dot: large+, small+, zero, small-, large-, NaN-prod, edge

    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    std::vector<double> pol_vec(static_cast<size_t>(N_stations * 6));
    std::vector<double> obs_pol(static_cast<size_t>(N_stations));

    // Set up pol_vec: each station has constant pol_vec = [1, 0, 0, 0, 0, 0]
    for (int s = 0; s < N_stations; ++s) {
        for (int c = 0; c < 6; ++c) pol_vec[s + c * N_stations] = 0.0;
        pol_vec[s + 0 * N_stations] = 1.0;  // only Mxx contributes
    }

    // Set up obs_pol
    obs_pol[0] =  1.0;   // expects positive
    obs_pol[1] =  1.0;   // expects positive, but small → still positive
    obs_pol[2] =  0.0;   // zero → skip
    obs_pol[3] =  0.0;   // zero → skip
    obs_pol[4] = -1.0;   // expects negative
    obs_pol[5] = -1.0;   // expects negative, but small → still negative
    obs_pol[6] = NAN;    // missing → NaN
    obs_pol[7] =  1.0;   // expects positive (last station for extra)

    // Set up mt: each trial gives different Mxx (dot = Mxx since pol_vec=[1,0,0,0,0,0])
    // Kernel uses mt[trial + comp * N_trials] (LayoutLeft [N_trials × 6])
    // Trial 0: large positive (Mxx=+5.0) → syn_pol=+1
    // Trial 1: small positive (Mxx=+0.001) → syn_pol=+1
    // Trial 2: zero (Mxx=0.0) → syn_pol=0
    // Trial 3: small negative (Mxx=-0.001) → syn_pol=-1
    // Trial 4: large negative (Mxx=-5.0) → syn_pol=-1
    // Trial 5: zero → syn_pol=0
    // Trial 6: large positive → syn_pol=+1
    double mt_raw[] = {
        5.0,  0.001,  0.0,  -0.001, -5.0,  0.0,  10.0,  // comp 0 (Mxx)
    };
    for (int t = 0; t < N_trials; ++t)
        mt[t + 0 * N_trials] = mt_raw[t];

    std::vector<double> misfit(static_cast<size_t>(N_stations * N_trials));

    fm::launch_polarity_kernel<Backend::OpenMP>(
        mt.data(), pol_vec.data(), obs_pol.data(), misfit.data(),
        N_stations, N_trials);

    int failures = 0;

    // Station 0: expects +1 → dot+ matches (trial 0,1,6), mismatch (trial 3,4)
    assert(!std::isnan(misfit[0 + 0 * N_stations]) && misfit[0 + 0 * N_stations] == 0.0);  // +1 vs +1 → match
    assert(!std::isnan(misfit[0 + 1 * N_stations]) && misfit[0 + 1 * N_stations] == 0.0);  // small+ vs +1 → match
    if (!std::isnan(misfit[0 + 3 * N_stations]) && misfit[0 + 3 * N_stations] != 1.0) ++failures;  // -1 vs +1 → mismatch

    // Station 2: expects 0 → zero dot → NaN (ambiguous zero/zero)
    if (!std::isnan(misfit[2 + 2 * N_stations])) {
        printf("  FAIL: station 2 trial 2 (zero/zero) should be NaN, got %g\n", misfit[2 + 2 * N_stations]);
        ++failures;
    }

    // Station 4: expects -1
    if (!std::isnan(misfit[4 + 4 * N_stations]) && misfit[4 + 4 * N_stations] != 0.0) ++failures;  // -5 vs -1 → match
    if (!std::isnan(misfit[4 + 0 * N_stations]) && misfit[4 + 0 * N_stations] != 1.0) ++failures;  // +5 vs -1 → mismatch

    // Station 6: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!std::isnan(misfit[6 + t * N_stations])) {
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

    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    std::vector<double> amp_P(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> amp_S(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> obs_psr(static_cast<size_t>(N_stations));

    // ---------- Setup ----------

    // mt: 3 trials with simple Mxx-only moment tensors
    // Kernel expects mt[trial + comp * N_trials]
    // Trial 0: Mxx = 1.0  (unit)
    // Trial 1: Mxx = 2.0  (amplitude ×2 → log10 ratio unchanged)
    // Trial 2: Mxx = 0.5  (amplitude ×0.5)
    mt[0 + 0 * N_trials] = 1.0;  // trial 0, comp 0
    mt[1 + 0 * N_trials] = 2.0;  // trial 1, comp 0
    mt[2 + 0 * N_trials] = 0.5;  // trial 2, comp 0

    // amp_P and amp_S: identity matrices with different scales
    // amp_P = diag(4.0) → √(mᵀ·4I·m) = √(4*mxx²) = 2*|mxx|
    // amp_S = diag(1.0) → √(mᵀ·1I·m) = √(mxx²) = |mxx|
    // P/S ratio = 2*|mxx| / |mxx| = 2.0, log10(2.0) ≈ 0.3010
    // This ratio is constant regardless of mxx scale (for diag)
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                amp_P[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? ((i == 0) ? 4.0 : 1e-30) : 0.0;
                amp_S[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? ((i == 0) ? 1.0 : 1e-30) : 0.0;
            }
        }
    }

    // obs_psr per station
    obs_psr[0] = std::log10(2.0);  // ~0.3010 — should match all trials
    obs_psr[1] = 1.0;                 // mismatched → misfit = (0.3010-1.0)²
    obs_psr[2] = 0.0;                 // mismatched
    obs_psr[3] = NAN;                 // missing → NaN

    std::vector<double> misfit(static_cast<size_t>(N_stations * N_trials));

    // ---------- Launch ----------
    fm::launch_psr_kernel<Backend::OpenMP>(
        mt.data(), amp_P.data(), amp_S.data(), obs_psr.data(), misfit.data(),
        N_stations, N_trials);

    int failures = 0;
    double expected_log = std::log10(2.0);  // ~0.30102999566

    // Station 0: obs = log10(2) ≈ 0.3010
    // Trial 0: syn_psr = log10(2*|1|/|1|) = log10(2) → misfit ≈ 0
    double m0 = misfit[0 + 0 * N_stations];
    if (!approx_eq(m0, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 0: expected ~0, got %.15g (diff=%.1e)\n",
               m0, m0 - 0.0);
        ++failures;
    }

    // Trial 1: syn_psr = log10(2*|2|/|2|) = log10(2) — same ratio → misfit ≈ 0
    double m1 = misfit[0 + 1 * N_stations];
    if (!approx_eq(m1, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 1: expected ~0, got %.15g\n", m1);
        ++failures;
    }

    // Trial 2: syn_psr = log10(2*|0.5|/|0.5|) = log10(2) — same
    double m2 = misfit[0 + 2 * N_stations];
    if (!approx_eq(m2, 0.0, 1e-12)) {
        printf("  FAIL: station 0 trial 2: expected ~0, got %.15g\n", m2);
        ++failures;
    }

    // Station 1: obs = 1.0 → misfit = (0.3010-1.0)² ≈ 0.4884
    double expected_diff = expected_log - 1.0;
    double expected_mis = expected_diff * expected_diff;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(misfit[1 + t * N_stations], expected_mis)) {
            printf("  FAIL: station 1 trial %d: expected %.10g, got %.10g\n",
                   t, expected_mis, misfit[1 + t * N_stations]);
            ++failures;
        }
    }

    // Station 2: obs = 0.0 → misfit = (0.3010)² ≈ 0.0906
    double expected_mis2 = expected_log * expected_log;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(misfit[2 + t * N_stations], expected_mis2)) {
            printf("  FAIL: station 2 trial %d: expected %.10g, got %.10g\n",
                   t, expected_mis2, misfit[2 + t * N_stations]);
            ++failures;
        }
    }

    // Station 3: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!std::isnan(misfit[3 + t * N_stations])) {
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

    const int N_stations = 1;
    const int N_trials = 2;

    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    std::vector<double> amp_P(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> amp_S(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> obs_psr(static_cast<size_t>(N_stations));
    std::vector<double> misfit(static_cast<size_t>(N_stations * N_trials));

    // mt trial 0: [1, 0, 0, 0, 0, 0] → Mxx only
    // mt trial 1: [0, 1, 0, 1, 0, 0] → Mxy only (note: row 0 col 1 for Index 3)
    for (int c = 0; c < 6; ++c) mt[0 + c * N_trials] = 0.0;
    mt[0 + 0 * N_trials] = 1.0;  // Mxx = 1
    for (int c = 0; c < 6; ++c) mt[1 + c * N_trials] = 0.0;
    mt[1 + 3 * N_trials] = 1.0;  // Mxy = 1 (index 3 in NED: [Mxx, Myy, Mzz, Mxy, Mxz, Myz])

    // amp_P: simple cross-terms between Mxx and Mxy
    // amp_P[0][0] = 4, amp_P[0][3] = amp_P[3][0] = 2, rest = 0
    // mᵀ·amp_P·m = Mxx²·4 + Mxy²·0 + 2·Mxx·Mxy·2 = 4·Mxx² + 4·Mxx·Mxy
    const int S = N_stations;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            amp_P[0 + i * S + j * (S * 6)] = 0.0;
    amp_P[0 + 0 * S + 0 * (S * 6)] = 4.0;   // Mxx·Mxx
    amp_P[0 + 0 * S + 3 * (S * 6)] = 2.0;   // Mxx·Mxy
    amp_P[0 + 3 * S + 0 * (S * 6)] = 2.0;   // Mxy·Mxx

    // amp_S: diag(1.0) — S-wave reference
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            amp_S[0 + i * S + j * (S * 6)] = (i == j) ? 1.0 : 0.0;

    obs_psr[0] = 0.0;  // expect log10(2/1) for trial 0

    fm::launch_psr_kernel<Backend::OpenMP>(
        mt.data(), amp_P.data(), amp_S.data(), obs_psr.data(), misfit.data(),
        N_stations, N_trials);

    int failures = 0;

    // Trial 0: Mxx=1, Mxy=0
    // quad_P = 4*1² = 4 → syn_amp_P = 2
    // quad_S = 1² = 1 → syn_amp_S = 1
    // syn_psr = log10(2/1) ≈ 0.3010, obs = 0 → misfit ≈ 0.0906
    double expected_syn_psr = std::log10(2.0);
    double expected_mis = expected_syn_psr * expected_syn_psr;
    if (!approx_eq(misfit[0 + 0 * N_stations], expected_mis)) {
        printf("  FAIL: trial 0 (Mxx only): expected %.10g, got %.10g\n",
               expected_mis, misfit[0 + 0 * N_stations]);
        ++failures;
    }

    // Trial 1: Mxy=1, Mxx=0
    // quad_P = 0 (no contributing diagonal/cross terms for Mxy alone with this amp_P)
    // Actually: Mxx=0, Mxy=1
    // quad_P = Mxx²·4 + Mxx·Mxy·2 + Mxy·Mxx·2 = 0
    // syn_amp_P = 0 → degenerate → NaN
    if (!std::isnan(misfit[0 + 1 * N_stations])) {
        printf("  FAIL: trial 1 (zero P amplitude): expected NaN, got %g\n", misfit[0 + 1 * N_stations]);
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

    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    std::vector<double> pol_vec(static_cast<size_t>(N_stations * 6));
    std::vector<double> obs_pol(static_cast<size_t>(N_stations));
    std::vector<double> misfit(static_cast<size_t>(N_stations * N_trials));

    // pol_vec: station 0 = [1,0,0,0,0,0], station 1 = [0,0,0,0,0,0], ...
    for (int s = 0; s < N_stations; ++s)
        for (int c = 0; c < 6; ++c) pol_vec[s + c * N_stations] = 0.0;
    pol_vec[0 + 0 * N_stations] = 1.0;   // non-zero pol
    // station 1: all zeros (missing polarity data)
    pol_vec[2 + 0 * N_stations] = -3.0;  // negative pol
    pol_vec[3 + 0 * N_stations] = 0.5;   // positive pol
    pol_vec[4 + 0 * N_stations] = 1.0;   // positive pol

    obs_pol[0] = 1.0;     // expects positive
    obs_pol[1] = 0.0;     // zero with all-zero pol_vec → skip
    obs_pol[2] = -1.0;    // expects negative
    obs_pol[3] = -1.0;    // expects negative, but pol is positive → mismatch
    obs_pol[4] = NAN;     // NaN obs → skip

    // Trial 0: Mxx=2 → dot positive for stations with non-zero pol[0]
    for (int c = 0; c < 6; ++c) mt[0 + c * N_trials] = 0.0;
    mt[0 + 0 * N_trials] = 2.0;
    // Trial 1: Mxx=-2 → dot negative
    for (int c = 0; c < 6; ++c) mt[1 + c * N_trials] = 0.0;
    mt[1 + 0 * N_trials] = -2.0;

    fm::launch_polarity_kernel<Backend::OpenMP>(
        mt.data(), pol_vec.data(), obs_pol.data(), misfit.data(),
        N_stations, N_trials);

    int failures = 0;

    // Station 0: obs=+1, pol[0]=1
    //   Trial 0: dot=2 → syn=+1 → match: 0.0
    if (misfit[0 + 0 * N_stations] != 0.0) { printf("  FAIL: s0 t0 expected 0, got %g\n", misfit[0 + 0 * N_stations]); ++failures; }
    //   Trial 1: dot=-2 → syn=-1 → mismatch: 1.0
    if (misfit[0 + 1 * N_stations] != 1.0) { printf("  FAIL: s0 t1 expected 1, got %g\n", misfit[0 + 1 * N_stations]); ++failures; }

    // Station 1: all-zero pol_vec, obs=0 → NaN
    if (!std::isnan(misfit[1 + 0 * N_stations])) { printf("  FAIL: s1 t0 expected NaN\n"); ++failures; }
    if (!std::isnan(misfit[1 + 1 * N_stations])) { printf("  FAIL: s1 t1 expected NaN\n"); ++failures; }

    // Station 2: obs=-1, pol[0]=-3
    //   Trial 0: dot=-6 → syn=-1 → match: 0.0
    if (misfit[2 + 0 * N_stations] != 0.0) { printf("  FAIL: s2 t0 expected 0, got %g\n", misfit[2 + 0 * N_stations]); ++failures; }
    //   Trial 1: dot=6 → syn=+1 → mismatch: 1.0
    if (misfit[2 + 1 * N_stations] != 1.0) { printf("  FAIL: s2 t1 expected 1, got %g\n", misfit[2 + 1 * N_stations]); ++failures; }

    // Station 3: obs=-1, pol[0]=0.5
    //   Trial 0: Mxx=2 → dot=+1.0 → syn=+1 → mismatch
    if (misfit[3 + 0 * N_stations] != 1.0) { printf("  FAIL: s3 t0 expected 1, got %g\n", misfit[3 + 0 * N_stations]); ++failures; }
    //   Trial 1: Mxx=-2 → dot=-1.0 → syn=-1 → match
    if (misfit[3 + 1 * N_stations] != 0.0) { printf("  FAIL: s3 t1 expected 0, got %g\n", misfit[3 + 1 * N_stations]); ++failures; }

    // Station 4: obs=NaN → all NaN
    if (!std::isnan(misfit[4 + 0 * N_stations])) { printf("  FAIL: s4 t0 expected NaN\n"); ++failures; }
    if (!std::isnan(misfit[4 + 1 * N_stations])) { printf("  FAIL: s4 t1 expected NaN\n"); ++failures; }

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

    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    std::vector<double> amp_P(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> amp_S(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> obs_psr(static_cast<size_t>(N_stations));
    std::vector<double> misfit(static_cast<size_t>(N_stations * N_trials));

    // mt: all Mxx=0
    for (int t = 0; t < N_trials; ++t)
        for (int c = 0; c < 6; ++c) mt[t + c * N_trials] = 0.0;
    mt[1 + 0 * N_trials] = 1e-20;  // ultra-tiny (should still compute, not be degenerate)

    // amp diag(1)
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                amp_P[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? 1.0 : 0.0;
                amp_S[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    obs_psr[0] = 0.0;   // log10(1/1) = 0
    obs_psr[1] = NAN;

    fm::launch_psr_kernel<Backend::OpenMP>(
        mt.data(), amp_P.data(), amp_S.data(), obs_psr.data(), misfit.data(),
        N_stations, N_trials);

    int failures = 0;

    // Station 0, Trial 0: all-zero mt → zero amplitude → NaN
    if (!std::isnan(misfit[0 + 0 * N_stations])) {
        printf("  FAIL: s0 t0 (zero mt) expected NaN, got %g\n", misfit[0 + 0 * N_stations]);
        ++failures;
    }

    // Station 0, Trial 1: ultra-tiny mt (1e-20)
    // sqrt(1e-40) = 1e-20 > 1e-30 threshold → not degenerate
    // syn_psr = log10(1e-20/1e-20) = 0.0, obs = 0.0 → misfit = 0.0² = 0.0
    if (!approx_eq(misfit[0 + 1 * N_stations], 0.0, 1e-12)) {
        printf("  FAIL: s0 t1 (tiny mt) expected 0, got %g\n", misfit[0 + 1 * N_stations]);
        ++failures;
    }

    // Station 1: NaN obs → all NaN
    if (!std::isnan(misfit[1 + 0 * N_stations])) { printf("  FAIL: s1 t0 expected NaN\n"); ++failures; }
    if (!std::isnan(misfit[1 + 1 * N_stations])) { printf("  FAIL: s1 t1 expected NaN\n"); ++failures; }

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
    std::vector<double> mt(static_cast<size_t>(N_trials * 6));
    for (int t = 0; t < N_trials; ++t)
        for (int c = 0; c < 6; ++c) mt[t + c * N_trials] = 0.0;
    mt[0 + 0 * N_trials] = 1.0;  // Mxx=1
    mt[1 + 0 * N_trials] = 2.0;  // Mxx=2
    mt[2 + 0 * N_trials] = -1.0; // Mxx=-1

    // ── pol_vec ──
    std::vector<double> pol_vec(static_cast<size_t>(N_stations * 6));
    for (int s = 0; s < N_stations; ++s)
        for (int c = 0; c < 6; ++c) pol_vec[s + c * N_stations] = 0.0;
    pol_vec[0 + 0 * N_stations] = 1.0;
    pol_vec[1 + 0 * N_stations] = 1.0;
    pol_vec[2 + 0 * N_stations] = 1.0;

    // ── obs_pol ──
    std::vector<double> obs_pol(static_cast<size_t>(N_stations));
    obs_pol[0] = 1.0;
    obs_pol[1] = -1.0;
    obs_pol[2] = 0.0;

    // ── amp_P, amp_S ──
    std::vector<double> amp_P(static_cast<size_t>(N_stations * 6 * 6));
    std::vector<double> amp_S(static_cast<size_t>(N_stations * 6 * 6));
    for (int s = 0; s < N_stations; ++s) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                amp_P[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? 4.0 : 0.0;
                amp_S[s + i * N_stations + j * (N_stations * 6)] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    // ── obs_psr ──
    std::vector<double> obs_psr(static_cast<size_t>(N_stations));
    obs_psr[0] = std::log10(2.0);
    obs_psr[1] = 0.5;
    obs_psr[2] = NAN;

    // ── misfit views ──
    std::vector<double> misfit_pol(static_cast<size_t>(N_stations * N_trials));
    std::vector<double> misfit_psr(static_cast<size_t>(N_stations * N_trials));

    // ── Launch back-to-back ──
    fm::launch_polarity_kernel<Backend::OpenMP>(
        mt.data(), pol_vec.data(), obs_pol.data(), misfit_pol.data(),
        N_stations, N_trials);
    fm::launch_psr_kernel<Backend::OpenMP>(
        mt.data(), amp_P.data(), amp_S.data(), obs_psr.data(), misfit_psr.data(),
        N_stations, N_trials);

    int failures = 0;

    // --- Verify some polarity outputs ---
    // Station 0: obs=+1, pol[0]=1, mt=[1, 2, -1] → syn=[+1,+1,-1]
    if (misfit_pol[0 + 0 * N_stations] != 0.0) { printf("  FAIL: pol s0 t0\n"); ++failures; }
    if (misfit_pol[0 + 1 * N_stations] != 0.0) { printf("  FAIL: pol s0 t1\n"); ++failures; }
    if (misfit_pol[0 + 2 * N_stations] != 1.0) { printf("  FAIL: pol s0 t2\n"); ++failures; }

    // Station 1: obs=-1 → syn=[+1,+1,-1] → [mismatch, mismatch, match]
    if (misfit_pol[1 + 0 * N_stations] != 1.0) { printf("  FAIL: pol s1 t0\n"); ++failures; }
    if (misfit_pol[1 + 2 * N_stations] != 0.0) { printf("  FAIL: pol s1 t2\n"); ++failures; }

    // --- Verify some PSR outputs ---
    // Station 0: obs=log10(2), amp_P=4I, amp_S=I
    //   syn_psr = log10(√4·|Mxx|² / √|Mxx|²) = log10(2·|Mxx|/|Mxx|) = log10(2)
    //   Independent of Mxx magnitude → misfit ≈ 0 for all trials
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(misfit_psr[0 + t * N_stations], 0.0, 1e-12)) {
            printf("  FAIL: psr s0 t%d: expected ~0, got %.15g\n", t, misfit_psr[0 + t * N_stations]);
            ++failures;
        }
    }

    // Station 1: obs=0.5
    //   syn_psr = log10(2) ≈ 0.3010
    //   misfit = (0.3010 - 0.5)² ≈ 0.0396
    double expected_mis = (std::log10(2.0) - 0.5);
    expected_mis *= expected_mis;
    for (int t = 0; t < N_trials; ++t) {
        if (!approx_eq(misfit_psr[1 + t * N_stations], expected_mis)) {
            printf("  FAIL: psr s1 t%d: expected %.10g, got %.10g\n", t, expected_mis, misfit_psr[1 + t * N_stations]);
            ++failures;
        }
    }

    // Station 2: NaN obs → all NaN
    for (int t = 0; t < N_trials; ++t) {
        if (!std::isnan(misfit_psr[2 + t * N_stations])) {
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
    (void)argc;
    (void)argv;

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

    return (total_failures == 0) ? 0 : 1;
}