#ifndef DATA_CACHE_H
#define DATA_CACHE_H

#include "backends/device.h"
#include <cstdint>
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ─ Trial struct — self-contained header, mirrors TrialSet HDF5 layout ─

struct Trial {
    // 1-based indices into /paraspace axes; physical values resolved below
    int32_t strike_idx;
    int32_t dip_idx;
    int32_t rake_idx;
    int32_t depth_idx;
    int32_t freq_idx;
    int32_t duration_idx;
    // Resolved physical values from /paraspace (angles in degrees, for MT)
    double strike;
    double dip;
    double rake;
};

using CacheKey = std::tuple<int, int, int>; // freq_idx, depth_idx, duration_idx

// ─ Per-module cache storage (flat double* arrays, no Kokkos::View) ─

struct XCorrCache {
    double *cc = nullptr;        // [n_phases × cc_rows × 6] column-major (per-lag obs·GF dots)
    double *synamp = nullptr;    // [n_phases × 36 × cc_rows] column-major (per-lag Gram)
    double *obs_norm2 = nullptr; // [n_phases]
    int cc_rows = 0;             // rows per phase = 2*maxlag+1
    int n_syn_phases = 0;        // phases * 6 (synamp row stride within one lag)
    int n_phases = 0;            // length of obs_norm2
    int maxlag = 0;              // half-width of the lag scan
};

struct WaveformCache {
    double *gf = nullptr; // [n_phases × n_samples × 6] column-major
    int n_phases = 0;
    int n_samples = 0;
};

struct PolarityCache {
    double *pol_vec = nullptr; // [n_phases × 6]  (pol_vec[phase + comp * n_phases])
    double *obs_pol = nullptr; // [n_phases]
    int n_phases = 0;
};

struct PSRCache {
    double *amp_P = nullptr;   // [n_phases × 6 × 6] (phase-major)
    double *amp_S = nullptr;   // [n_phases × 6 × 6]
    double *obs_psr = nullptr; // [n_phases]
    int n_phases = 0;
};

// ─ Cache entry keyed by (freq_idx, depth_idx, duration_idx) ─

struct CacheEntry {
    int freq_idx;
    int depth_idx;
    int duration_idx;
    int maxlag;
    int n_phases;
    int n_stations;

    XCorrCache xcorr;
    WaveformCache waveform;
    PolarityCache polarity;
    PSRCache psr;

    CacheEntry()
        : freq_idx(-1), depth_idx(-1), duration_idx(-1), maxlag(0), n_phases(0), n_stations(0) {
    }

    bool valid() const {
        return freq_idx >= 0;
    }
    void release() {
        delete[] xcorr.cc;
        delete[] xcorr.synamp;
        delete[] xcorr.obs_norm2;
        delete[] waveform.gf;
        delete[] polarity.pol_vec;
        delete[] polarity.obs_pol;
        delete[] psr.amp_P;
        delete[] psr.amp_S;
        delete[] psr.obs_psr;
        xcorr = XCorrCache();
        waveform = WaveformCache();
        polarity = PolarityCache();
        psr = PSRCache();
    }
};

// ─ DataCache — host-side data cache for forward.cpp ─

class DataCache {
  public:
    /// Construct with a maxlag value for XCorr precomputation.
    explicit DataCache(int maxlag, bool retain_waveforms = false, std::string p_module = "XcorrP",
                       std::string s_module = "XcorrS");

    /// Load all (freq_idx, depth_idx, duration_idx) combos referenced by trials from
    /// database.h5.
    /// @param database_path  Path to database.h5 (HDF5)
    /// @param trials         Trial parameters extracted from status_{N}.h5
    void load_from_database(const std::string &database_path, const std::vector<Trial> &trials);

    /// Retrieve or compute cached entry.
    /// Returns a const pointer — caller must not modify cached data.
    const CacheEntry *get_or_compute(int freq_idx, int depth_idx, int duration_idx);

    /// Free all memory held by the cache.
    void release_all();

    /// Number of cached entries.
    size_t size() const {
        return cache_.size();
    }

    /// Maximum lag (XCorr window).
    int maxlag() const {
        return maxlag_;
    }

  private:
    struct CacheKeyHash {
        size_t operator()(const CacheKey &key) const {
            const auto [freq_idx, depth_idx, duration_idx] = key;
            return (static_cast<size_t>(freq_idx) * 31 + static_cast<size_t>(depth_idx)) * 31 +
                   static_cast<size_t>(duration_idx);
        }
    };
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> cache_;

    int maxlag_;
    bool retain_waveforms_;
    std::string p_module_;
    std::string s_module_;

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Extract unique (freq_idx, depth_idx, duration_idx) combos from trial set.
    static std::vector<CacheKey> unique_combos(const std::vector<Trial> &trials);

    /// Read all preprocessed data for one (freq_idx, depth_idx, duration_idx) combo
    /// from database.h5 and compute reductions.
    CacheEntry load_combo(const std::string &database_path, int freq_idx, int depth_idx,
                          int duration_idx, const std::vector<std::string> &phase_ids,
                          int n_stations, int n_p, int n_s);

    /// XCorr reduction: compute CC, synamp, obs_norm2.
    static void compute_xcorr_reduction(CacheEntry &entry, const std::vector<double> &obs,
                                        const std::vector<double> &gf, int n_samples);

    /// Polarity reduction: sum gf_pol over time into pol_vec.
    static void compute_polarity_reduction(CacheEntry &entry, const std::vector<double> &gf_pol,
                                           int n_pol_samples);

    /// PSR reduction: compute amp_P, amp_S from GF matrices.
    static void compute_psr_reduction(CacheEntry &entry, const std::vector<double> &ampP_host,
                                      const std::vector<double> &ampS_host,
                                      const std::vector<double> &obs_psr_host);

    /// Read string 1D dataset from HDF5 (phase_ids).
    static std::vector<std::string> read_phase_ids(hid_t file_id, const char *path);

    /// Read vector of int 1D from HDF5 (station_idx).
    static std::vector<int> read_int_1d_direct(hid_t file_id, const char *path);
};

#endif // DATA_CACHE_H
