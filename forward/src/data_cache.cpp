#include "data_cache.h"
#include "hdf5_io.h"

#include <iostream>
#include <set>
#include <sstream>
#include <cmath>
#include <cstring>
#include <cstdlib>

// ──────────────────────────────────────────────────────────────────────────
// DataCache construction
// ──────────────────────────────────────────────────────────────────────────

DataCache::DataCache(int maxlag)
    : maxlag_(maxlag)
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
        if (buf[i]) {
            result.push_back(std::string(buf[i]));
            std::free(buf[i]);
        }
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

    // ── Allocate flat arrays ──────────────────────────────────────────────

    if (has_xcorr) {
        entry.xcorr.cc_rows = n_ph * (2 * maxlag_ + 1);  // cc: [cc_rows × 6]
        entry.xcorr.cc = new double[static_cast<size_t>(entry.xcorr.cc_rows * 6)];
        entry.xcorr.n_syn_phases = n_ph * 6;             // synamp: [n_ph*6 × 6]
        entry.xcorr.synamp = new double[static_cast<size_t>(entry.xcorr.n_syn_phases * 6)];
        entry.xcorr.n_phases = n_ph;
        entry.xcorr.obs_norm2 = new double[n_ph];
    }

    if (has_polarity) {
        entry.polarity.n_phases = n_ph;
        entry.polarity.pol_vec = new double[static_cast<size_t>(n_ph * 6)];
        entry.polarity.obs_pol = new double[n_ph];
    }

    if (has_psr) {
        entry.psr.n_phases = n_ph;
        entry.psr.amp_P = new double[static_cast<size_t>(n_ph * 6 * 6)];
        entry.psr.amp_S = new double[static_cast<size_t>(n_ph * 6 * 6)];
        entry.psr.obs_psr = new double[n_ph];
    }

    // ── Compute reductions per phase (directly into flat arrays) ──────────

    // XCorr — per phase: compute CC, synamp, obs_norm2
    if (has_xcorr) {
        double* cc_total   = entry.xcorr.cc;
        double* synamp_tot = entry.xcorr.synamp;
        double* obs_norm2  = entry.xcorr.obs_norm2;
        int cc_rows = 2 * maxlag_ + 1;
        int n_ph_big = n_ph;  // column stride for synamp: [n_ph × 36]

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            if (hd.n_xcorr == 0) {
                obs_norm2[i] = 0.0;
                continue;
            }

            // obs_norm2 = sum(obs^2)
            double norm2 = 0.0;
            for (int j = 0; j < hd.n_xcorr; ++j) {
                norm2 += hd.obs[j] * hd.obs[j];
            }
            obs_norm2[i] = norm2;

            // synamp[6][6] = gf^T * gf
            // stored as [n_ph*6 × 6] column-major: synamp_tot[i*6 + a + b * (n_ph*6)]
            int N = hd.n_xcorr;
            for (int a = 0; a < 6; ++a) {
                for (int b = a; b < 6; ++b) {
                    double sum = 0.0;
                    for (int t = 0; t < N; ++t) {
                        sum += hd.gf[t * 6 + a] * hd.gf[t * 6 + b];
                    }
                    synamp_tot[i * 6 + a + b * (n_ph * 6)] = sum;
                    synamp_tot[i * 6 + b + a * (n_ph * 6)] = sum;  // symmetric
                }
            }

            // CC[2*maxlag+1][6] — time-domain cross-correlation
            // stored as [n_ph*(2*maxlag+1) × 6] column-major
            int maxlag = maxlag_;
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
                    // column-major: row = i * cc_rows + lag_idx, col = comp
                    cc_total[i * cc_rows + lag_idx + comp * (n_ph * cc_rows)] = sum;
                }
            }
        }
    }

    // Polarity — per phase: sum gf_pol over time → pol_vec[6]
    if (has_polarity) {
        double* pol_vec_flat = entry.polarity.pol_vec;
        double* obs_pol_flat = entry.polarity.obs_pol;

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            if (hd.n_pol == 0) {
                for (int c = 0; c < 6; ++c) pol_vec_flat[i + c * n_ph] = 0.0;
                obs_pol_flat[i] = 0.0;
                continue;
            }

            // Sum gf_pol over time axis for each component
            // Layout: pol_vec[phase + comp * n_ph] (column-major [n_ph × 6])
            for (int c = 0; c < 6; ++c) {
                double sum = 0.0;
                for (int t = 0; t < hd.n_pol; ++t) {
                    sum += hd.gf_pol[t * 6 + c];
                }
                pol_vec_flat[i + c * n_ph] = sum;
            }

            obs_pol_flat[i] = static_cast<double>(hd.obs_pol);
        }
    }

    // PSR — per phase: copy precomputed amp_P, amp_S, obs_psr
    if (has_psr) {
        double* ampP_f = entry.psr.amp_P;
        double* ampS_f = entry.psr.amp_S;
        double* obs_f  = entry.psr.obs_psr;

        for (int i = 0; i < n_ph; ++i) {
            auto& hd = host_data[i];
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    double valP = (!hd.ampP.empty()) ? hd.ampP[a * 6 + b] : 0.0;
                    double valS = (!hd.ampS.empty()) ? hd.ampS[a * 6 + b] : 0.0;
                    // Layout: amp[phase + a*n_ph + b*(n_ph*6)]
                    ampP_f[i + a * n_ph + b * (n_ph * 6)] = valP;
                    ampS_f[i + a * n_ph + b * (n_ph * 6)] = valS;
                }
            }
            obs_f[i] = hd.obs_psr;
        }
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
// Static reduction helpers (stubs — actual reductions are inline above)
// ──────────────────────────────────────────────────────────────────────────

void DataCache::compute_xcorr_reduction(CacheEntry& /*entry*/,
                                         const std::vector<double>& /*obs*/,
                                         const std::vector<double>& /*gf*/,
                                         int /*n_samples*/) {
    // Reductions are performed inline in load_combo().
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
    // See load_combo() — PSR data is precomputed in database.h5, copied directly.
}