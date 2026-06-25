#ifndef DATA_CACHE_H
#define DATA_CACHE_H

#include "backends/device.h"
#include <cstdint>
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────
// Trial struct — replicated here for self-contained header.
// Mirrors the TrialSet layout in HDF5.
// ──────────────────────────────────────────────────────────────────────────

struct Trial {
  double strike;
  double dip;
  double rake;
  double depth;
  int32_t depth_idx;
  int32_t freq_idx;
};

// ──────────────────────────────────────────────────────────────────────────
// Per-module cache storage types (flat double* arrays — no Kokkos::View)
// ──────────────────────────────────────────────────────────────────────────

struct XCorrCache {
  double *cc = nullptr; // [cc_rows × 6]  (flattened across phases)
  double *synamp =
      nullptr; // [n_syn_phases × 6 × 6]  (same layout as Kokkos LayoutLeft)
  double *obs_norm2 = nullptr; // [n_phases]
  int cc_rows = 0;             // total rows in cc = n_phases * (2*maxlag+1)
  int n_syn_phases = 0;        // rows in synamp = n_phases * 6
  int n_phases = 0;            // length of obs_norm2
};

struct PolarityCache {
  double *pol_vec =
      nullptr; // [n_phases × 6]  (pol_vec[phase + comp * n_phases])
  double *obs_pol = nullptr; // [n_phases]
  int n_phases = 0;
};

struct PSRCache {
  double *amp_P = nullptr;   // [n_phases × 6 × 6]  (amp_P[phase + i*n_phases +
                             // j*(n_phases*6)])
  double *amp_S = nullptr;   // [n_phases × 6 × 6]
  double *obs_psr = nullptr; // [n_phases]
  int n_phases = 0;
};

// ──────────────────────────────────────────────────────────────────────────
// Single cache entry for a (freq_idx, depth_idx) key.
// Contains host-resident reduced data for all phases/stations.
// ──────────────────────────────────────────────────────────────────────────

struct CacheEntry {
  int freq_idx;
  int depth_idx;
  int maxlag;
  int n_phases;
  int n_stations;

  XCorrCache xcorr;
  PolarityCache polarity;
  PSRCache psr;

  CacheEntry()
      : freq_idx(-1), depth_idx(-1), maxlag(0), n_phases(0), n_stations(0) {}

  bool valid() const { return freq_idx >= 0; }
  void release() {
    delete[] xcorr.cc;
    delete[] xcorr.synamp;
    delete[] xcorr.obs_norm2;
    delete[] polarity.pol_vec;
    delete[] polarity.obs_pol;
    delete[] psr.amp_P;
    delete[] psr.amp_S;
    delete[] psr.obs_psr;
    xcorr = XCorrCache();
    polarity = PolarityCache();
    psr = PSRCache();
  }
};

// ──────────────────────────────────────────────────────────────────────────
// DataCache — host-side data cache for forward.cpp
// ──────────────────────────────────────────────────────────────────────────

class DataCache {
public:
  /// Construct with a maxlag value for XCorr precomputation.
  explicit DataCache(int maxlag);

  /// Load all (freq_idx, depth_idx) combos referenced by trials from
  /// database.h5.
  /// @param database_path  Path to database.h5 (HDF5)
  /// @param trials         Trial parameters extracted from status_{N}.h5
  void load_from_database(const std::string &database_path,
                          const std::vector<Trial> &trials);

  /// Retrieve or compute cached entry.
  /// Returns a const pointer — caller must not modify cached data.
  const CacheEntry *get_or_compute(int freq_idx, int depth_idx);

  /// Free all memory held by the cache.
  void release_all();

  /// Number of cached entries.
  size_t size() const { return cache_.size(); }

  /// Maximum lag (XCorr window).
  int maxlag() const { return maxlag_; }

private:
  // Cache: (freq_idx, depth_idx) → CacheEntry
  // Using pair of ints as key with a simple hash.
  struct PairHash {
    size_t operator()(const std::pair<int, int> &p) const {
      return static_cast<size_t>(p.first) * 31 + static_cast<size_t>(p.second);
    }
  };
  std::unordered_map<std::pair<int, int>, CacheEntry, PairHash> cache_;

  int maxlag_;

  // ── Internal helpers ──────────────────────────────────────────────────

  /// Extract unique (freq_idx, depth_idx) combos from trial set.
  static std::vector<std::pair<int, int>>
  unique_combos(const std::vector<Trial> &trials);

  /// Read all preprocessed data for one (freq_idx, depth_idx) combo
  /// from database.h5 and compute reductions.
  CacheEntry load_combo(const std::string &database_path, int freq_idx,
                        int depth_idx,
                        const std::vector<std::string> &phase_ids,
                        int n_stations);

  /// XCorr reduction: compute CC, synamp, obs_norm2.
  static void compute_xcorr_reduction(CacheEntry &entry,
                                      const std::vector<double> &obs,
                                      const std::vector<double> &gf,
                                      int n_samples);

  /// Polarity reduction: sum gf_pol over time into pol_vec.
  static void compute_polarity_reduction(CacheEntry &entry,
                                         const std::vector<double> &gf_pol,
                                         int n_pol_samples);

  /// PSR reduction: compute amp_P, amp_S from GF matrices.
  static void compute_psr_reduction(CacheEntry &entry,
                                    const std::vector<double> &ampP_host,
                                    const std::vector<double> &ampS_host,
                                    const std::vector<double> &obs_psr_host);

  /// Read string 1D dataset from HDF5 (phase_ids).
  static std::vector<std::string> read_phase_ids(hid_t file_id,
                                                 const char *path);

  /// Read vector of int 1D from HDF5 (station_idx).
  static std::vector<int> read_int_1d_direct(hid_t file_id, const char *path);
};

#endif // DATA_CACHE_H