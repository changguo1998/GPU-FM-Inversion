#ifndef DATA_CACHE_H
#define DATA_CACHE_H

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <stdexcept>

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
// Per-module cache storage types
// ──────────────────────────────────────────────────────────────────────────

struct XCorrCache {
    // cc[k][i]  — cross-correlation, shape [2*maxlag+1] × 6
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> cc;
    // synamp — GF auto-correlation matrix [6][6]
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> synamp;
    // obs_norm2 — ‖obs‖² per phase [N_phases]
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> obs_norm2;
};

struct PolarityCache {
    // pol_vec[6] per station — sum of gf_pol over time
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> pol_vec;
    // obs_pol — int8 converted to double: -1, 0, +1
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> obs_pol;
};

struct PSRCache {
    // amp_P[6][6], amp_S[6][6] per station
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> amp_P;
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> amp_S;
    // obs_psr per station
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> obs_psr;
};

// ──────────────────────────────────────────────────────────────────────────
// Single cache entry for a (freq_idx, depth_idx) key.
// Contains GPU-resident reduced data for all phases/stations.
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

    CacheEntry() : freq_idx(-1), depth_idx(-1), maxlag(0), n_phases(0), n_stations(0) {}

    bool valid() const { return freq_idx >= 0; }
    void release() {
        xcorr.cc = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>();
        xcorr.synamp = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>();
        xcorr.obs_norm2 = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>();
        polarity.pol_vec = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>();
        polarity.obs_pol = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>();
        psr.amp_P = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>();
        psr.amp_S = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>();
        psr.obs_psr = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>();
    }
};

// ──────────────────────────────────────────────────────────────────────────
// DataCache — GPU data cache for forward.cpp
// ──────────────────────────────────────────────────────────────────────────

class DataCache {
public:
    /// Construct with a maxlag value for XCorr precomputation.
    explicit DataCache(int maxlag);

    /// Load all (freq_idx, depth_idx) combos referenced by trials from database.h5.
    /// @param database_path  Path to database.h5 (HDF5)
    /// @param trials         Trial parameters extracted from status_{N}.h5
    void load_from_database(const std::string& database_path,
                            const std::vector<Trial>& trials);

    /// Retrieve or compute cached entry.
    /// Returns a const pointer — caller must not modify GPU data.
    const CacheEntry* get_or_compute(int freq_idx, int depth_idx);

    /// Free all GPU memory held by the cache.
    void release_all();

    /// Number of cached entries.
    size_t size() const { return cache_.size(); }

    /// Maximum lag (XCorr window).
    int maxlag() const { return maxlag_; }

private:
    // Kokkos execution space
    Kokkos::DefaultExecutionSpace exec_space_;

    // Cache: (freq_idx, depth_idx) → CacheEntry
    // Using pair of ints as key with a simple hash.
    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const {
            return static_cast<size_t>(p.first) * 31 + static_cast<size_t>(p.second);
        }
    };
    std::unordered_map<std::pair<int,int>, CacheEntry, PairHash> cache_;

    int maxlag_;

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Extract unique (freq_idx, depth_idx) combos from trial set.
    static std::vector<std::pair<int,int>> unique_combos(const std::vector<Trial>& trials);

    /// Read all preprocessed data for one (freq_idx, depth_idx) combo
    /// from database.h5 and compute GPU reductions.
    CacheEntry load_combo(const std::string& database_path,
                          int freq_idx, int depth_idx,
                          const std::vector<std::string>& phase_ids,
                          int n_stations);

    /// XCorr GPU reduction: compute CC, synamp, obs_norm2.
    static void compute_xcorr_reduction(CacheEntry& entry,
                                        const std::vector<double>& obs,
                                        const std::vector<double>& gf,
                                        int n_samples);

    /// Polarity GPU reduction: sum gf_pol over time into pol_vec.
    static void compute_polarity_reduction(CacheEntry& entry,
                                           const std::vector<double>& gf_pol,
                                           int n_pol_samples);

    /// PSR GPU reduction: compute amp_P, amp_S from GF matrices.
    static void compute_psr_reduction(CacheEntry& entry,
                                      const std::vector<double>& ampP_host,
                                      const std::vector<double>& ampS_host,
                                      const std::vector<double>& obs_psr_host);

    /// Read string 1D dataset from HDF5 (phase_ids).
    /// This needs HDF5 C API directly since Hdf5Handle doesn't have string read.
    static std::vector<std::string> read_phase_ids(hid_t file_id, const char* path);

    /// Read vector of int 1D from HDF5 (station_idx).
    static std::vector<int> read_int_1d_direct(hid_t file_id, const char* path);
};

#endif // DATA_CACHE_H