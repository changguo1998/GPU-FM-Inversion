/// Test for XCorr misfit kernel (OpenMP backend).
///
/// Verifies:
///  1. Hand-computed reference match for small synthetic data
///  2. Gram matrix identity: mᵀ·synamp·m = ‖GF·m‖²
///  3. Boundary: zero-norm, maxlag=0, single phase, single trial

#include "backends/device.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "kernels/xcorr_kernel.h"

// helpers
static inline double sqr(double x) { return x * x; }
static inline bool approx(double a, double b, double tol = 1e-10) {
  double d = (a > b) ? (a - b) : (b - a);
  return d < tol;
}

// ────────────────────────────────────────────────────────────────
// Compute misfit manually (pure C++) for one (p, t)
// Data layout: column-major matching xcorr_kernel.h
//   mt[trial + comp * N_trials], synamp[phase + (i*6+j) * Nph]
//   cc[(phase*cc_pp+k) + i*(Nph*cc_pp)]
// ────────────────────────────────────────────────────────────────
double ref_misfit(int phase, int trial, const double *mt, const double *cc_data,
                  const double *synamp_data, const double *obs_norm2, int Ntr,
                  int Nph, int cc_pp) {
  double m[6];
  for (int c = 0; c < 6; ++c)
    m[c] = mt[trial + c * Ntr];

  // syn_norm2: mᵀ·synamp·m
  double syn_n2 = 0.0;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      syn_n2 += m[i] * synamp_data[phase + (i * 6 + j) * Nph] * m[j];

  if (syn_n2 <= 0.0 || obs_norm2[phase] <= 0.0)
    return 1.0;

  double denom = std::sqrt(obs_norm2[phase] * syn_n2);

  double best = 0.0;
  for (int k = 0; k < cc_pp; ++k) {
    double cc = 0.0;
    for (int i = 0; i < 6; ++i)
      cc += m[i] * cc_data[(phase * cc_pp + k) + i * (Nph * cc_pp)];
    double v = std::fabs(cc / denom);
    if (v > best)
      best = v;
  }
  return 1.0 - best;
}

int main() {
  int passed = 0, total = 0;
  printf("=== XCorr Kernel Tests ===\n\n");

  // ── Test 1: Hand-computed reference (2 phases, 3 trials, maxlag=1) ──
  {
    const int Nph = 2, Ntr = 3, cc_pp = 3; // maxlag=1 → 2*1+1 = 3

    // Column-major: mt[trial + comp * Ntr], cc[(p*cc_pp+k) + i*(Nph*cc_pp)]
    std::vector<double> h_mt(6 * Ntr);
    std::vector<double> h_cc(Nph * cc_pp * 6);
    std::vector<double> h_syn(Nph * 36);
    std::vector<double> h_obs(Nph);
    std::vector<double> h_mis(Nph * Ntr);

    srand(42);

    for (int p = 0; p < Nph; ++p)
      for (int k = 0; k < cc_pp; ++k)
        for (int i = 0; i < 6; ++i)
          h_cc[(p * cc_pp + k) + i * (Nph * cc_pp)] =
              (double)(rand() % 1000) / 1000.0 - 0.2;

    for (int p = 0; p < Nph; ++p)
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          h_syn[p + (i * 6 + j) * Nph] = (i == j)
                                             ? (1.0 + (rand() % 100) / 500.0)
                                             : ((rand() % 100) / 1000.0 - 0.05);

    h_obs[0] = 3.7;
    h_obs[1] = 2.1;

    for (int t = 0; t < Ntr; ++t) {
      double norm = 0.0;
      for (int c = 0; c < 6; ++c) {
        double v = (rand() % 2000) / 1000.0 - 1.0;
        h_mt[t + c * Ntr] = v;
        norm += v * v;
      }
      norm = std::sqrt(norm);
      for (int c = 0; c < 6; ++c)
        h_mt[t + c * Ntr] /= norm;
    }

    fm::launch_xcorr_misfit<Backend::OpenMP>(h_mt.data(), h_cc.data(),
                                             h_syn.data(), h_obs.data(),
                                             h_mis.data(), Nph, Ntr, cc_pp);

    bool ok = true;
    double max_err = 0.0;
    for (int p = 0; p < Nph; ++p) {
      for (int t = 0; t < Ntr; ++t) {
        double expected =
            ref_misfit(p, t, h_mt.data(), h_cc.data(), h_syn.data(),
                       h_obs.data(), Ntr, Nph, cc_pp);
        double diff = std::fabs(h_mis[p + t * Nph] - expected);
        if (diff > max_err)
          max_err = diff;
        if (diff > 1e-10) {
          ok = false;
        }
      }
    }
    total++;
    if (ok) {
      passed++;
      printf("  [PASS] Test 1: Reference match  (max diff = %e)\n", max_err);
    } else
      printf("  [FAIL] Test 1: Reference match  (max diff = %e)\n", max_err);
  }

  // ── Test 2: Gram matrix identity ──
  //   Build GF, synamp, random m. Verify mᵀ·synamp·m = ‖GF·m‖².
  //   Results captured for test 3.
  {
    const int Nsamp = 20;
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

    // Random m, normalized
    double m[6] = {0.3, 0.1, -0.5, 0.8, -0.2, 0.4};
    double mn = 0;
    for (int i = 0; i < 6; ++i)
      mn += m[i] * m[i];
    mn = std::sqrt(mn);
    for (int i = 0; i < 6; ++i)
      m[i] /= mn;

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
      printf("  [PASS] Test 2: Gram identity   lhs=%e  rhs=%e  diff=%e\n", lhs,
             rhs, std::fabs(lhs - rhs));
    } else {
      printf("  [FAIL] Test 2: Gram identity   lhs=%e  rhs=%e  diff=%e\n", lhs,
             rhs, std::fabs(lhs - rhs));
    }

    // ── Test 3: Kernel with Gram-consistent synamp ──
    // Reuse syn, m, lhs from test 2
    {
      const int Nph = 1, Ntr = 1, cc_pp = 1;

      std::vector<double> g_mt(6 * Ntr);
      std::vector<double> g_cc(Nph * cc_pp * 6);
      std::vector<double> g_syn(Nph * 36);
      std::vector<double> g_obs(Nph);
      std::vector<double> g_mis(Nph * Ntr);

      for (int k = 0; k < 36; ++k)
        g_syn[0 + k * Nph] = syn[k];
      g_obs[0] = 1.0;
      for (int i = 0; i < 6; ++i)
        g_cc[0 + i * (Nph * cc_pp)] = 1.0;
      for (int i = 0; i < 6; ++i)
        g_mt[0 + i * Ntr] = m[i];

      fm::launch_xcorr_misfit<Backend::OpenMP>(g_mt.data(), g_cc.data(),
                                               g_syn.data(), g_obs.data(),
                                               g_mis.data(), Nph, Ntr, cc_pp);

      double expected_sum = 0.0;
      for (int i = 0; i < 6; ++i)
        expected_sum += m[i];

      double misfit_gold = 1.0 - std::fabs(expected_sum / std::sqrt(lhs));

      total++;
      if (approx(g_mis[0], misfit_gold, 1e-12)) {
        passed++;
        printf("  [PASS] Test 3: End-to-end      misfit=%e  expected=%e\n",
               g_mis[0], misfit_gold);
      } else {
        printf("  [FAIL] Test 3: End-to-end      misfit=%e  expected=%e\n",
               g_mis[0], misfit_gold);
      }
    }
  }

  // ── Test 4: Boundary — zero syn_norm2 (should return 1.0) ──
  {
    const int Nph = 1, Ntr = 1, cc_pp = 1;

    std::vector<double> z_mt(6 * Ntr);
    std::vector<double> z_cc(Nph * cc_pp * 6);
    std::vector<double> z_syn(Nph * 36);
    std::vector<double> z_obs(Nph);
    std::vector<double> z_mis(Nph * Ntr);

    for (int k = 0; k < 36; ++k)
      z_syn[0 + k * Nph] = 0.0;
    z_obs[0] = 0.0;
    z_cc[0 + 0 * (Nph * cc_pp)] = 1.0;
    for (int i = 1; i < 6; ++i)
      z_cc[0 + i * (Nph * cc_pp)] = 0.0;
    for (int i = 0; i < 6; ++i)
      z_mt[0 + i * Ntr] = 0.0;

    fm::launch_xcorr_misfit<Backend::OpenMP>(z_mt.data(), z_cc.data(),
                                             z_syn.data(), z_obs.data(),
                                             z_mis.data(), Nph, Ntr, cc_pp);

    total++;
    if (z_mis[0] == 1.0) {
      passed++;
      printf("  [PASS] Test 4: Zero-norm guard    misfit=%.1f\n", z_mis[0]);
    } else {
      printf("  [FAIL] Test 4: Zero-norm guard    misfit=%e (expected 1.0)\n",
             z_mis[0]);
    }
  }

  // ── Test 5: Boundary — single phase, single trial, maxlag=0 (cc_pp=1) ──
  {
    const int Nph = 1, Ntr = 1, cc_pp = 1;

    std::vector<double> s_mt(6 * Ntr);
    std::vector<double> s_cc(Nph * cc_pp * 6);
    std::vector<double> s_syn(Nph * 36);
    std::vector<double> s_obs(Nph);
    std::vector<double> s_mis(Nph * Ntr);

    for (int k = 0; k < 6; ++k)
      s_mt[0 + k * Ntr] = 0.0;
    s_mt[0 + 0 * Ntr] = 1.0;

    s_cc[0 + 0 * (Nph * cc_pp)] = 1.0;
    for (int i = 1; i < 6; ++i)
      s_cc[0 + i * (Nph * cc_pp)] = 0.0;

    for (int k = 0; k < 36; ++k)
      s_syn[0 + k * Nph] = 0.0;
    s_syn[0 + 0 * Nph] = 1.0;

    s_obs[0] = 1.0;

    fm::launch_xcorr_misfit<Backend::OpenMP>(s_mt.data(), s_cc.data(),
                                             s_syn.data(), s_obs.data(),
                                             s_mis.data(), Nph, Ntr, cc_pp);

    total++;
    if (approx(s_mis[0], 0.0, 1e-12)) {
      passed++;
      printf("  [PASS] Test 5: Minimal 1×1×1       misfit=%e\n", s_mis[0]);
    } else {
      printf("  [FAIL] Test 5: Minimal 1×1×1       misfit=%e (expected 0)\n",
             s_mis[0]);
    }
  }

  printf("\n=== %d/%d tests passed ===\n", passed, total);
  return (passed != total) ? 1 : 0;
}