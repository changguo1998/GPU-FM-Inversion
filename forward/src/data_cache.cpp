#include "data_cache.h"
#include "hdf5_io.h"

#include <iostream>
#include <set>
#include <sstream>
#include <cmath>
#include <cstring>

// ──────────────────────────────────────────────────────────────────────────
// DataCache construction
// ──────────────────────────────────────────────────────────────────────────

DataCache::DataCache(int maxlag)
    : exec_space_(Kokkos::DefaultExecutionSpace())
    , maxlag_(maxlag)
{}

// ──────────────────────────────────────────────────────────────────────────
// Helper: extract unique (freq, depth) combos
// ──────────────────────────────────────────────────────────────────────────

std::vector<std::pair<int,int>>
DataCache::unique_combos(const std::vector<Trial>& trials) {
    std::set<std::pair<int,int>> seen;
    for (const auto& t : trials) {
        seen.insert({t.freq_idx, t.depth_idx});
    }
    return std::vector<std::pair<int,int>>(seen.begin(), seen.end());
}

// ──────────────────────────────────────────────────────────────────────────
// Helper: read phase_ids from HDF5 index (string array)
// ──────────────────────────────────────────────────────────────────────────

std::vector<std::string> DataCache::read_phase_ids(hid_t file_id, const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open " + std::string(path));

    hid_t dtype = H5Dget_type(dset);
    if (H5Tget_class(dtype) != H5T_STRING)
        throw std::runtime_error(path + std::string(": not a string dataset"));
    H5Tclose(dtype);

    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

    // Read as variable-length strings
    std::vector<char*> buf(dims[0]);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, H5T_VARIABLE);
    H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

    std::vector<std::string> result;
    for (hsize_t i = 0; i < dims[0]; ++i) {
        result.push_back(std::string(buf[i]));
        std::free(buf[i]);
    }
    H5Tclose(memtype);
    H5Sclose(space);
    H5Dclose(dset);
    return result;
}

// ──────────────────────────────────────────────────────────────────────────
// Helper: read int 1D from HDF5
// ──────────────────────────────────────────────────────────────────────────

std::vector<int> DataCache::read_int_1d_direct(hid_t file_id, const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open " + std::string(path));

    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

    std::vector<int> result(dims[0]);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());

    H5Sclose(space);
    H5Dclose(dset);
    return result;
}

// ──────────────────────────────────────────────────────────────────────────
// load_from_database
// ──────────────────────────────────────────────────────────────────────────

void DataCache::load_from_database(const std::string& database_path,
                                    const std::vector<Trial>& trials) {
    // 1. Find unique combos
    auto combos = unique_combos(trials);
    if (combos.empty()) {
        std::cerr << "DataCache: no (freq, depth) combos in trials" << std::endl;
        return;
    }

    // 2. Open database.h5 and read index
    hid_t file_id = H5Fopen(database_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) throw std::runtime_error("Cannot open " + database_path);

    std::vector<std::string> phase_ids;
    int n_stations = 0;
    try {
        phase_ids = read_phase_ids(file_id, "/index/phase_ids");
        // Count unique stations
        std::set<std::string> station_set;
        for (const auto& pid : phase_ids) {
            // phase_id format: "NET.STA.COMP.TYPE"
            size_t dot1 = pid.find('.');
            if (dot1 != std::string::npos) {
                size_t dot2 = pid.find('.', dot1 + 1);
                if (dot2 != std::string::npos) {
                    station_set.insert(pid.substr(0, dot2));
                } else {
                    station_set.insert(pid);
                }
            }
        }
        n_stations = static_cast<int>(station_set.size());
    } catch (...) {
        H5Fclose(file_id);
        throw;
    }

    // 3. Load each combo
    try {
        for (const auto& combo : combos) {
            int freq_idx = combo.first;
            int depth_idx = combo.second;

            // Skip if already cached
            if (cache_.find(combo) != cache_.end()) continue;

            CacheEntry entry = load_combo(database_path, freq_idx, depth_idx,
                                          phase_ids, n_stations);
            cache_[combo] = std::move(entry);
        }
    } catch (...) {
        H5Fclose(file_id);
        throw;
    }

    H5Fclose(file_id);
}

// ──────────────────────────────────────────────────────────────────────────
// load_combo — read data for one (freq, depth) combo and reduce
// ──────────────────────────────────────────────────────────────────────────

CacheEntry DataCache::load_combo(const std::string& database_path,
                                  int freq_idx, int depth_idx,
                                  const std::vector<std::string>& phase_ids,
                                  int n_stations) {
    CacheEntry entry;
    entry.freq_idx = freq_idx;
    entry.depth_idx = depth_idx;
    entry.maxlag = maxlag_;
    entry.n_phases = static_cast<int>(phase_ids.size());
    entry.n_stations = n_stations;

    Hdf5Handle h5;
    h5.open(database_path.c_str(), H5F_ACC_RDONLY);

    std::string freq_str = std::to_string(freq_idx);
    std::string depth_str = std::to_string(depth_idx);

    int n_ph = entry.n_phases;

    // ── Collect host-side data per phase ──────────────────────────────────
    struct PhaseHostData {
        std::vector<double> obs;      // XCorr: [N]
        std::vector<double> gf;       // XCorr: [N * 6]
        std::vector<double> gf_pol;   // Polarity: [N_pol * 6]
        std::vector<double> ampP;     // PSR: [6 * 6]
        std::vector<double> ampS;     // PSR: [6 * 6]
        double obs_psr = 0.0;
        int obs_pol = 0;
        int n_xcorr = 0;
        int n_pol = 0;
    };

    std::vector<PhaseHostData> host_data(n_ph);
    bool has_xcorr = false;
    bool has_polarity = false;
    bool has_psr = false;

    for (int i = 0; i < n_ph; ++i) {
        const std::string& pid = phase_ids[i];
        std::string prefix = "/data/" + freq_str + "/";

        // ── XCorr ─────────────────────────────────────────────────────────
        std::string xcorr_path = prefix + "XCorr/" + pid + "/";
        try {
            std::string obs_path = xcorr_path + "obs";
            if (h5.group_exists(obs_path.c_str())) {
                host_data[i].obs = h5.read_double_1d(obs_path.c_str());
                host_data[i].n_xcorr = static_cast<int>(host_data[i].obs.size());
                host_data[i].gf = h5.read_double_1d((xcorr_path + "gf").c_str());
                has_xcorr = true;
            }
        } catch (...) { /* module not present for this phase */ }

        // ── Polarity ──────────────────────────────────────────────────────
        std::string pol_path = prefix + "Polarity/" + pid + "/";
        try {
            std::string gf_pol_path = pol_path + "gf_pol";
            if (h5.group_exists(gf_pol_path.c_str())) {
                int r, c;
                host_data[i].gf_pol = h5.read_double_2d(gf_pol_path.c_str(), r, c);
                host_data[i].n_pol = r;
                host_data[i].obs_pol = h5.read_int_scalar((pol_path + "obs_pol").c_str());
                has_polarity = true;
            }
        } catch (...) { /* module not present */ }

        // ── PSR ───────────────────────────────────────────────────────────
        std::string psr_path = prefix + "PSR/" + pid + "/";
        try {
            std::string ampP_path = psr_path + "amp_P";
            if (h5.group_exists(ampP_path.c_str())) {
                int r, c;
                host_data[i].ampP = h5.read_double_2d(ampP_path.c_str(), r, c);
                host_data[i].ampS = h5.read_double_2d((psr_path + "amp_S").c_str(), r, c);
                host_data[i].obs_psr = h5.read_double_scalar((psr_path + "obs_psr").c_str());
                has_psr = true;
            }
        } catch (...) { /* module not present */ }
    }

    h5.close();

    // ── Allocate GPU views ────────────────────────────────────────────────

    if (has_xcorr) {
        entry.xcorr.cc = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>(
            "xcorr_cc", 2 * maxlag_ + 1, 6);
        entry.xcorr.synamp = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>(
            "xcorr_synamp", 6, 6);
        entry.xcorr.obs_norm2 = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>(
            "xcorr_obs_norm2", n_ph);
    }

    if (has_polarity) {
        entry.polarity.pol_vec = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>(
            "polarity_pol_vec", n_ph);  // pol_vec[6] flattened across phases
        entry.polarity.obs_pol = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>(
            "polarity_obs_pol", n_ph);
    }

    if (has_psr) {
        entry.psr.amp_P = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>(
            "psr_amp_P", n_ph * 6, 6);
        entry.psr.amp_S = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>(
            "psr_amp_S", n_ph * 6, 6);
        entry.psr.obs_psr = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>(
            "psr_obs_psr", n_ph);
    }

    // ── Perform Kokkos reductions per phase ───────────────────────────────
    // Note: In a full implementation, we'd batch these across phases.
    // For correctness, we process each phase sequentially.

    // XCorr — per phase: compute CC, synamp, obs_norm2
    if (has_xcorr) {
        // Accumulate views for all phases into one flat layout
        // cc_total: [n_ph * (2*maxlag+1), 6]
        Kokkos::View<double**, Kokkos::DefaultExecutionSpace> cc_total(
            "xcorr_cc_total", n_ph * (2 * maxlag_ + 1), 6);
        Kokkos::View<double**, Kokkos::DefaultExecutionSpace> synamp_total(
            "xcorr_synamp_total", n_ph * 6, 6);

        auto cc_total_h = Kokkos::create_mirror_view(cc_total);
        auto synamp_total_h = Kokkos::create_mirror_view(synamp_total);
        auto obs_norm2_h = Kokkos::create_mirror_view(entry.xcorr.obs_norm2);

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            if (hd.n_xcorr == 0) {
                obs_norm2_h(i) = 0.0;
                continue;
            }

            // obs_norm2 = sum(obs^2)
            double norm2 = 0.0;
            for (int j = 0; j < hd.n_xcorr; ++j) {
                norm2 += hd.obs[j] * hd.obs[j];
            }
            obs_norm2_h(i) = norm2;

            // synamp[6][6] = gf^T * gf
            int N = hd.n_xcorr;
            for (int a = 0; a < 6; ++a) {
                for (int b = a; b < 6; ++b) {
                    double sum = 0.0;
                    for (int t = 0; t < N; ++t) {
                        sum += hd.gf[t * 6 + a] * hd.gf[t * 6 + b];
                    }
                    synamp_total_h(i * 6 + a, b) = sum;
                    synamp_total_h(i * 6 + b, a) = sum;  // symmetric
                }
            }

            // CC[2*maxlag+1][6] — time-domain cross-correlation
            int maxlag = maxlag_;
            int cc_rows = 2 * maxlag + 1;
            for (int lag = -maxlag; lag <= maxlag; ++lag) {
                int lag_idx = lag + maxlag;
                for (int comp = 0; comp < 6; ++comp) {
                    double sum = 0.0;
                    for (int t = 0; t < N; ++t) {
                        int t_shift = t + lag;
                        if (t_shift >= 0 && t_shift < N) {
                            sum += hd.obs[t_shift] * hd.gf[t * 6 + comp];
                        }
                    }
                    cc_total_h(i * cc_rows + lag_idx, comp) = sum;
                }
            }
        }

        Kokkos::deep_copy(cc_total, cc_total_h);
        Kokkos::deep_copy(synamp_total, synamp_total_h);
        Kokkos::deep_copy(entry.xcorr.obs_norm2, obs_norm2_h);

        entry.xcorr.cc = cc_total;
        entry.xcorr.synamp = synamp_total;
    }

    // Polarity — per phase: sum gf_pol over time → pol_vec[6]
    if (has_polarity) {
        // pol_vec layout: [n_ph * 6] flattened
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> pol_vec_total(
            "polarity_pol_vec_total", n_ph * 6);

        auto pol_vec_h = Kokkos::create_mirror_view(pol_vec_total);
        auto obs_pol_h = Kokkos::create_mirror_view(entry.polarity.obs_pol);

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            if (hd.n_pol == 0) {
                for (int c = 0; c < 6; ++c) pol_vec_h(i * 6 + c) = 0.0;
                obs_pol_h(i) = 0.0;
                continue;
            }

            // Sum gf_pol over time axis for each component
            for (int c = 0; c < 6; ++c) {
                double sum = 0.0;
                for (int t = 0; t < hd.n_pol; ++t) {
                    sum += hd.gf_pol[t * 6 + c];
                }
                pol_vec_h(i * 6 + c) = sum;
            }

            obs_pol_h(i) = static_cast<double>(hd.obs_pol);
        }

        Kokkos::deep_copy(pol_vec_total, pol_vec_h);
        Kokkos::deep_copy(entry.polarity.obs_pol, obs_pol_h);

        entry.polarity.pol_vec = pol_vec_total;
    }

    // PSR — per phase: copy precomputed amp_P, amp_S, obs_psr to GPU
    if (has_psr) {
        auto ampP_h = Kokkos::create_mirror_view(entry.psr.amp_P);
        auto ampS_h = Kokkos::create_mirror_view(entry.psr.amp_S);
        auto obs_psr_h = Kokkos::create_mirror_view(entry.psr.obs_psr);

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    double valP = (!hd.ampP.empty()) ? hd.ampP[a * 6 + b] : 0.0;
                    double valS = (!hd.ampS.empty()) ? hd.ampS[a * 6 + b] : 0.0;
                    ampP_h(i * 6 + a, b) = valP;
                    ampS_h(i * 6 + a, b) = valS;
                }
            }
            obs_psr_h(i) = hd.obs_psr;
        }

        Kokkos::deep_copy(entry.psr.amp_P, ampP_h);
        Kokkos::deep_copy(entry.psr.amp_S, ampS_h);
        Kokkos::deep_copy(entry.psr.obs_psr, obs_psr_h);
    }

    return entry;
}

// ──────────────────────────────────────────────────────────────────────────
// get_or_compute
// ──────────────────────────────────────────────────────────────────────────

const CacheEntry* DataCache::get_or_compute(int freq_idx, int depth_idx) {
    auto key = std::make_pair(freq_idx, depth_idx);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return &it->second;
    }
    throw std::runtime_error(
        "DataCache: combo (" + std::to_string(freq_idx) + ", " +
        std::to_string(depth_idx) + ") not loaded");
}

// ──────────────────────────────────────────────────────────────────────────
// release_all
// ──────────────────────────────────────────────────────────────────────────

void DataCache::release_all() {
    for (auto& kv : cache_) {
        kv.second.release();
    }
    cache_.clear();
}

// ──────────────────────────────────────────────────────────────────────────
// Static reduction helpers (for documentation; actual reductions inline above)
// ──────────────────────────────────────────────────────────────────────────

void DataCache::compute_xcorr_reduction(CacheEntry& /*entry*/,
                                         const std::vector<double>& /*obs*/,
                                         const std::vector<double>& /*gf*/,
                                         int /*n_samples*/) {
    // Reductions are performed inline in load_combo().
    // This method is reserved for future refactoring into a Kokkos parallel_reduce.
}

void DataCache::compute_polarity_reduction(CacheEntry& /*entry*/,
                                            const std::vector<double>& /*gf_pol*/,
                                            int /*n_pol_samples*/) {
    // See load_combo() — inline host-side reduction.
}

void DataCache::compute_psr_reduction(CacheEntry& /*entry*/,
                                       const std::vector<double>& /*ampP_host*/,
                                       const std::vector<double>& /*ampS_host*/,
                                       const std::vector<double>& /*obs_psr_host*/) {
    // See load_combo() — PSR data is precomputed in database.h5, just copied to GPU.
}