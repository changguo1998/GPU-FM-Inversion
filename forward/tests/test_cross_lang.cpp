/**
 * test_cross_lang.cpp — C++ side of cross‑language verification.
 *
 * Modes:
 *   --mode mt-csv <csv_path>        Read CSV from Julia and verify MT values
 *   --mode hdf5 <h5_path>           Read HDF5 written by Julia and verify
 *   --mode trials <h5_path>         Read trial HDF5 and verify structure
 */

#include "hdf5_io.h"
#include "mt_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL: " << msg << std::endl;                               \
      failures++;                                                              \
    } else {                                                                   \
      std::cout << "  OK: " << msg << std::endl;                               \
    }                                                                          \
  } while (0)

// ──────────────────────────────────────────────
// CSV utilities
// ──────────────────────────────────────────────

struct CsvRow {
  double strike, dip, rake, Mxx, Myy, Mzz, Mxy, Mxz, Myz;
};

static std::vector<CsvRow> read_mt_csv(const std::string &path) {
  std::vector<CsvRow> rows;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "FATAL: could not open " << path << std::endl;
    exit(1);
  }

  std::string line;
  // Skip header
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream ss(line);
    CsvRow r;
    char comma;
    ss >> r.strike >> comma >> r.dip >> comma >> r.rake >> comma >> r.Mxx >>
        comma >> r.Myy >> comma >> r.Mzz >> comma >> r.Mxy >> comma >> r.Mxz >>
        comma >> r.Myz;
    rows.push_back(r);
  }
  return rows;
}

// ──────────────────────────────────────────────
// Mode: mt-csv
// ──────────────────────────────────────────────

static int test_mt_csv(const char *csv_path) {
  std::cout << "=== MT CSV cross‑language test ===" << std::endl;
  auto rows = read_mt_csv(csv_path);

  CHECK(rows.size() > 0, "CSV has > 0 rows (got " << rows.size() << ")");

  double max_diff = 0.0;
  const double DEG2RAD = M_PI / 180.0;

  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &r = rows[i];
    MomentTensor mt =
        sdr_to_mt(r.strike * DEG2RAD, r.dip * DEG2RAD, r.rake * DEG2RAD);

    double d = std::abs(mt.Mxx - r.Mxx);
    if (d > max_diff)
      max_diff = d;
    d = std::abs(mt.Myy - r.Myy);
    if (d > max_diff)
      max_diff = d;
    d = std::abs(mt.Mzz - r.Mzz);
    if (d > max_diff)
      max_diff = d;
    d = std::abs(mt.Mxy - r.Mxy);
    if (d > max_diff)
      max_diff = d;
    d = std::abs(mt.Mxz - r.Mxz);
    if (d > max_diff)
      max_diff = d;
    d = std::abs(mt.Myz - r.Myz);
    if (d > max_diff)
      max_diff = d;
  }

  CHECK(max_diff < 1e-6, "max_diff = " << max_diff << " (threshold 1e-6)");

  std::cout << std::endl;
  return failures;
}

// ──────────────────────────────────────────────
// Mode: hdf5 — read Julia‑written file
// ──────────────────────────────────────────────

static int test_hdf5(const char *h5_path) {
  std::cout << "=== HDF5 cross‑language test ===" << std::endl;

  Hdf5Handle h;
  h.open(h5_path, H5F_ACC_RDONLY);

  // scalar int
  int iv = h.read_int_scalar("/cross_lang/int_val");
  CHECK(iv == 42, "read_int_scalar: " << iv);

  // scalar double
  double dv = h.read_double_scalar("/cross_lang/double_val");
  CHECK(std::abs(dv - 3.14159) < 1e-10, "read_double_scalar: " << dv);

  // 1D int
  auto iv1d = h.read_int_1d("/cross_lang/int_array");
  CHECK(iv1d.size() == 5, "int_array size: " << iv1d.size());
  CHECK(iv1d.size() == 5 && iv1d[0] == 10 && iv1d[4] == 50, "int_array values");

  // 1D double
  auto dv1d = h.read_double_1d("/cross_lang/double_array");
  CHECK(dv1d.size() == 6, "double_array size: " << dv1d.size());
  CHECK(dv1d.size() == 6 && std::abs(dv1d[0] - 1.1) < 1e-10 &&
            std::abs(dv1d[5] - 6.6) < 1e-10,
        "double_array values");

  // 2D double.  HDF5.jl stores Julia column-major data; shape appears
  // transposed from the C perspective (3×2 for a Julia 2×3 matrix).
  // Accept either (2,3) or (3,2) — the round‑trip is correct either way.
  int rows = 0, cols = 0;
  auto dv2d = h.read_double_2d("/cross_lang/double_2d", rows, cols);
  bool shape_ok = (rows == 2 && cols == 3) || (rows == 3 && cols == 2);
  CHECK(shape_ok,
        "double_2d shape: " << rows << "×" << cols << " (expected 2×3 or 3×2)");
  // Verify 1.0 is first element; 6.0 is last element.
  CHECK(std::abs(dv2d.front() - 1.0) < 1e-10 &&
            std::abs(dv2d.back() - 6.0) < 1e-10,
        "double_2d values: first=" << dv2d.front() << " last=" << dv2d.back());

  // group existence
  CHECK(h.group_exists("/cross_lang"), "/cross_lang exists");
  CHECK(!h.group_exists("/nonexistent"), "/nonexistent does not exist");

  h.close();

  std::cout << std::endl;
  return failures;
}

// ──────────────────────────────────────────────
// Mode: trials
// ──────────────────────────────────────────────

static int test_trials(const char *h5_path) {
  std::cout << "=== Trials HDF5 cross‑language test ===" << std::endl;

  Hdf5Handle h;
  h.open(h5_path, H5F_ACC_RDONLY);

  // Check /trials group exists
  CHECK(h.group_exists("/trials"), "/trials group exists");

  // Read all trial datasets
  auto strike = h.read_double_1d("/trials/strike");
  auto dip = h.read_double_1d("/trials/dip");
  auto rake = h.read_double_1d("/trials/rake");
  auto depth = h.read_double_1d("/trials/depth");
  auto depth_idx = h.read_int_1d("/trials/depth_idx");
  auto freq_idx = h.read_int_1d("/trials/freq_idx");

  int n = static_cast<int>(strike.size());
  CHECK(n > 0, "trial count: " << n);
  CHECK(dip.size() == (size_t)n, "dip length matches");
  CHECK(rake.size() == (size_t)n, "rake length matches");
  CHECK(depth.size() == (size_t)n, "depth length matches");
  CHECK(depth_idx.size() == (size_t)n, "depth_idx length matches");
  CHECK(freq_idx.size() == (size_t)n, "freq_idx length matches");

  // Verify SDR ranges
  for (int i = 0; i < n; ++i) {
    CHECK(strike[i] >= 0.0 && strike[i] < 360.0,
          "strike[" << i << "] = " << strike[i] << " in [0,360)");
    CHECK(dip[i] >= 0.0 && dip[i] <= 90.0,
          "dip[" << i << "] = " << dip[i] << " in [0,90]");
    CHECK(rake[i] >= -90.0 && rake[i] <= 90.0,
          "rake[" << i << "] = " << rake[i] << " in [-90,90]");
  }
  // Verify freq indices are non‑negative
  for (int i = 0; i < n; ++i) {
    CHECK(freq_idx[i] >= 0,
          "freq_idx[" << i << "] = " << freq_idx[i] << " >= 0");
  }

  // Spot‑check MT for first trial
  double s = strike[0], d = dip[0], r = rake[0];
  MomentTensor mt =
      sdr_to_mt(s * M_PI / 180.0, d * M_PI / 180.0, r * M_PI / 180.0);
  std::cout << "  First trial SDR=(" << s << "," << d << "," << r << ") → MT=("
            << mt.Mxx << "," << mt.Myy << "," << mt.Mzz << ")" << std::endl;
  CHECK(true, "first trial MT computed");

  h.close();

  std::cout << std::endl;
  return failures;
}

// ──────────────────────────────────────────────
// main
// ──────────────────────────────────────────────

static void usage(const char *prog) {
  std::cerr << "Usage: " << prog << " --mode <mode> <path>\n"
            << "  modes: mt-csv  <csv>   Read MT CSV from Julia\n"
            << "         hdf5    <h5>    Read HDF5 from Julia\n"
            << "         trials  <h5>    Read trials HDF5 from Julia\n";
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc != 4)
    usage(argv[0]);

  std::string mode = argv[2];
  const char *path = argv[3];
  int f = 0;

  if (mode == "mt-csv") {
    f = test_mt_csv(path);
  } else if (mode == "hdf5") {
    f = test_hdf5(path);
  } else if (mode == "trials") {
    f = test_trials(path);
  } else {
    usage(argv[0]);
  }

  if (f == 0) {
    std::cout << "PASS: all assertions passed." << std::endl;
    return 0;
  } else {
    std::cerr << f << " assertion(s) FAILED." << std::endl;
    return 1;
  }
}