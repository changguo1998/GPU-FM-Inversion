#include "data_cache.h"
#include "hdf5_io.h"
#include "validation.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>

// ─ DataCache construction ─

DataCache::DataCache(int maxlag) : maxlag_(maxlag) {
}

// ─ Helper: extract unique (freq, depth, duration) combos ─

std::vector<CacheKey> DataCache::unique_combos(const std::vector<Trial> &trials) {
    std::set<CacheKey> seen;
    for (const auto &t : trials) {
        seen.insert({t.freq_idx, t.depth_idx, t.duration_idx});
    }
    return std::vector<CacheKey>(seen.begin(), seen.end());
}

// ─ Read phase_ids from HDF5 index ─

std::vector<std::string> DataCache::read_phase_ids(hid_t file_id, const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    if (dset < 0)
        throw std::runtime_error("Cannot open " + std::string(path));

    hid_t dtype = H5Dget_type(dset);
    if (H5Tget_class(dtype) != H5T_STRING)
        throw std::runtime_error(path + std::string(": not a string dataset"));
    H5Tclose(dtype);

    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

    // Read as variable-length strings
    std::vector<char *> buf(dims[0]);
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

// ─ Read int 1D dataset from HDF5 ─

std::vector<int> DataCache::read_int_1d_direct(hid_t file_id, const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    if (dset < 0)
        throw std::runtime_error("Cannot open " + std::string(path));

    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

    std::vector<int> result(dims[0]);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());

    H5Sclose(space);
    H5Dclose(dset);
    return result;
}

// ─ load_from_database ─

void DataCache::load_from_database(const std::string &database_path,
                                   const std::vector<Trial> &trials) {
    // 1. Find unique combos
    auto combos = unique_combos(trials);
    if (combos.empty())
        throw std::runtime_error("DataCache: no (freq, depth, duration) combos in trials");

    // 2. Open database.h5 and read index
    hid_t file_id = H5Fopen(database_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Cannot open " + database_path);

    std::vector<std::string> phase_ids;
    std::vector<std::string> p_ids;
    std::vector<std::string> s_ids;
    int n_stations = 0;
    try {
        // Read channel_id from each group (P then S) -- each group optional
        if (Hdf5Handle::link_exists(file_id, "/XcorrP/channel_id"))
            p_ids = read_phase_ids(file_id, "/XcorrP/channel_id");
        if (Hdf5Handle::link_exists(file_id, "/XcorrS/channel_id"))
            s_ids = read_phase_ids(file_id, "/XcorrS/channel_id");
        // Combine: P first, S after (matches xcorr array convention)
        phase_ids.reserve(p_ids.size() + s_ids.size());
        phase_ids.insert(phase_ids.end(), p_ids.begin(), p_ids.end());
        phase_ids.insert(phase_ids.end(), s_ids.begin(), s_ids.end());
        // Count unique stations from channel_id strings
        std::set<std::string> station_set;
        for (const auto &cid : phase_ids) {
            // channel_id format: "NET.STA.CHAN" - extract station part
            size_t dot = cid.find('.');
            if (dot != std::string::npos) {
                size_t dot2 = cid.find('.', dot + 1);
                if (dot2 != std::string::npos)
                    station_set.insert(cid.substr(0, dot2));
                else
                    station_set.insert(cid);
            }
        }
        n_stations = static_cast<int>(station_set.size());
    } catch (...) {
        H5Fclose(file_id);
        throw;
    }

    // 3. Load each combo
    try {
        for (const auto &combo : combos) {
            const auto [freq_idx, depth_idx, duration_idx] = combo;

            // Skip if already cached
            if (cache_.find(combo) != cache_.end())
                continue;

            CacheEntry entry =
                load_combo(database_path, freq_idx, depth_idx, duration_idx, phase_ids, n_stations,
                           static_cast<int>(p_ids.size()), static_cast<int>(s_ids.size()));
            cache_[combo] = std::move(entry);
        }
    } catch (...) {
        H5Fclose(file_id);
        throw;
    }

    H5Fclose(file_id);
}

// ─ load_combo: read + reduce one (freq, depth) combo ─

CacheEntry DataCache::load_combo(const std::string &database_path, int freq_idx, int depth_idx,
                                 int duration_idx, const std::vector<std::string> &phase_ids,
                                 int n_stations, int n_p, int n_s) {
    CacheEntry entry;
    entry.freq_idx = freq_idx;
    entry.depth_idx = depth_idx;
    entry.duration_idx = duration_idx;
    entry.maxlag = maxlag_;
    entry.n_phases = static_cast<int>(phase_ids.size());
    entry.n_stations = n_stations;

    Hdf5Handle h5;
    h5.open(database_path.c_str(), H5F_ACC_RDONLY);

    std::string freq_str = std::to_string(freq_idx);
    std::string duration_str = std::to_string(duration_idx);
    std::string depth_str; // 1-based GF depth index (matches /paraspace/depth, /trials/depth_idx)

    int n_ph = entry.n_phases;

    // ── Collect host-side data per phase ──────────────────────────────────
    struct PhaseHostData {
        std::vector<double> obs;    // XCorr: [N]
        std::vector<double> gf;     // XCorr: [N * 6]
        std::vector<double> gf_pol; // Polarity: [N_pol * 6]
        std::vector<double> ampP;   // PSR: [6 * 6]
        std::vector<double> ampS;   // PSR: [6 * 6]
        double obs_psr = 0.0;
        int obs_pol = 0;
        int n_xcorr = 0;
        int n_pol = 0;
    };

    std::vector<PhaseHostData> host_data(n_ph);
    bool has_xcorr = false;
    bool has_polarity = false;
    bool has_psr = false;

    // GF group names use the 1-based depth index directly (no float formatting).
    depth_str = std::to_string(depth_idx);

    // ── Read station_idx for phase->channel mapping (polarity) ─────────────
    // Combined from XcorrP (P phases first) then xcorrS (S phases)
    std::vector<int> station_idx;
    std::vector<int> p_si, s_si;
    if (Hdf5Handle::link_exists(h5.file_id, "/XcorrP/station_idx"))
        p_si = h5.read_int_1d("/XcorrP/station_idx");
    if (Hdf5Handle::link_exists(h5.file_id, "/XcorrS/station_idx"))
        s_si = h5.read_int_1d("/XcorrS/station_idx");
    station_idx.reserve(p_si.size() + s_si.size());
    station_idx.insert(station_idx.end(), p_si.begin(), p_si.end());
    station_idx.insert(station_idx.end(), s_si.begin(), s_si.end());
    if (station_idx.size() != static_cast<size_t>(n_ph))
        throw std::runtime_error("DataCache: station_idx length does not match phase count");

    // ── Determine P/S indices (data already partitioned in groups) ──────
    std::vector<int> p_indices(n_p), s_indices(n_s);
    for (int i = 0; i < n_p; ++i)
        p_indices[i] = i;
    for (int i = 0; i < n_s; ++i)
        s_indices[i] = n_p + i;

    // ── Read XCorr data from new schema ───────────────────────────────────

    // Process P phases
    if (n_p > 0) {
        const std::string obs_path = "/XcorrP/obs/" + freq_str + "/obs";
        if (!h5.group_exists(obs_path.c_str()))
            throw std::runtime_error("DataCache: missing " + obs_path);
        // Read obs: [N_samples, N_phases_P]
        int n_obs, n_ph_p;
        std::vector<double> obs_p = h5.read_double_2d(obs_path.c_str(), n_obs, n_ph_p);
        if (n_obs <= 0 || n_ph_p != n_p)
            throw std::runtime_error("DataCache: invalid shape for " + obs_path);

        // Read GF: [N_samples, 6, N_phases_P]
        std::string gf_path =
            "/XcorrP/gf/" + depth_str + "/" + freq_str + "/" + duration_str + "/gf";
        if (!h5.group_exists(gf_path.c_str()))
            throw std::runtime_error("DataCache: missing " + gf_path);
        int n_gf, n_comp, n_ph_gf;
        std::vector<double> gf_p = h5.read_double_3d(gf_path.c_str(), n_gf, n_comp, n_ph_gf);
        if (n_gf != n_obs || n_comp != 6 || n_ph_gf != n_p)
            throw std::runtime_error("DataCache: invalid shape for " + gf_path);

        for (int j = 0; j < n_p; ++j) {
            int i = p_indices[j];
            auto &hd = host_data[i];
            // obs: column-major in file is row-major C order: [n_obs, n_p]
            // Column j is at offsets: j, j+n_p, j+2*n_p, ...
            hd.obs.resize(n_obs);
            for (int t = 0; t < n_obs; ++t)
                hd.obs[t] = obs_p[t * n_p + j];
            hd.n_xcorr = n_obs;

            // gf: [n_obs, 6, n_p] in C order
            // For phase j, component c, time t:
            //   offset = t*6*n_p + c*n_p + j
            hd.gf.resize(fm::checked_mul(static_cast<size_t>(n_obs), 6, "P GF phase"));
            for (int t = 0; t < n_obs; ++t)
                for (int c = 0; c < 6; ++c)
                    hd.gf[t * 6 + c] = gf_p[t * 6 * n_p + c * n_p + j];
            has_xcorr = true;
        }
    }

    // Process S phases (same approach)
    if (n_s > 0) {
        const std::string obs_path = "/XcorrS/obs/" + freq_str + "/obs";
        if (!h5.group_exists(obs_path.c_str()))
            throw std::runtime_error("DataCache: missing " + obs_path);
        int n_obs, n_ph_s;
        std::vector<double> obs_s = h5.read_double_2d(obs_path.c_str(), n_obs, n_ph_s);
        if (n_obs <= 0 || n_ph_s != n_s)
            throw std::runtime_error("DataCache: invalid shape for " + obs_path);

        std::string gf_path =
            "/XcorrS/gf/" + depth_str + "/" + freq_str + "/" + duration_str + "/gf";
        if (!h5.group_exists(gf_path.c_str()))
            throw std::runtime_error("DataCache: missing " + gf_path);
        int n_gf, n_comp, n_ph_gf;
        std::vector<double> gf_s = h5.read_double_3d(gf_path.c_str(), n_gf, n_comp, n_ph_gf);
        if (n_gf != n_obs || n_comp != 6 || n_ph_gf != n_s)
            throw std::runtime_error("DataCache: invalid shape for " + gf_path);

        for (int j = 0; j < n_s; ++j) {
            int i = s_indices[j];
            auto &hd = host_data[i];
            hd.obs.resize(n_obs);
            for (int t = 0; t < n_obs; ++t)
                hd.obs[t] = obs_s[t * n_s + j];
            hd.n_xcorr = n_obs;

            hd.gf.resize(fm::checked_mul(static_cast<size_t>(n_obs), 6, "S GF phase"));
            for (int t = 0; t < n_obs; ++t)
                for (int c = 0; c < 6; ++c)
                    hd.gf[t * 6 + c] = gf_s[t * 6 * n_s + c * n_s + j];
            has_xcorr = true;
        }
    }

    // ── Read Polarity data from new schema ────────────────────────────────
    if (h5.group_exists("/PolarityP/obs")) {
        // Obs: [N_channels] — map per-channel to per-phase via station_idx
        std::vector<double> pol_obs_all;
        try {
            pol_obs_all = h5.read_double_1d("/PolarityP/obs/1/obs");
        } catch (...) {
        }

        // GF: [N_pol_samples, 6, N_channels]
        std::string pol_gf_path = "/PolarityP/gf/" + depth_str + "/1/gf";
        int n_pol_samp = 0, n_pol_comp = 0, n_pol_ch = 0;
        std::vector<double> pol_gf_all;
        if (h5.group_exists(pol_gf_path.c_str())) {
            try {
                pol_gf_all =
                    h5.read_double_3d(pol_gf_path.c_str(), n_pol_samp, n_pol_comp, n_pol_ch);
            } catch (...) {
            }
        }
        bool pol_gf_ok = (!pol_gf_all.empty() && n_pol_comp == 6);

        if (!pol_obs_all.empty()) {
            has_polarity = true;
            // Map each phase to its polarity channel via station_idx (1-based)
            for (int i = 0; i < n_ph; ++i) {
                auto &hd = host_data[i];
                int ch =
                    (station_idx[i] >= 1 && station_idx[i] <= static_cast<int>(pol_obs_all.size()))
                        ? station_idx[i] - 1
                        : -1;
                if (ch >= 0) {
                    hd.obs_pol = static_cast<int>(pol_obs_all[ch]);
                    if (pol_gf_ok && n_pol_ch > ch) {
                        hd.n_pol = n_pol_samp;
                        hd.gf_pol.resize(n_pol_samp * 6);
                        for (int t = 0; t < n_pol_samp; ++t)
                            for (int c = 0; c < 6; ++c)
                                hd.gf_pol[t * 6 + c] =
                                    pol_gf_all[t * 6 * n_pol_ch + c * n_pol_ch + ch];
                    }
                }
            }
        }
    }

    h5.close();

    if (!has_xcorr || n_ph == 0)
        throw std::runtime_error("DataCache: combo has no XCorr data");

    int window_length = -1;
    for (int i = 0; i < n_ph; ++i) {
        const auto &hd = host_data[i];
        if (hd.n_xcorr <= 0 || hd.obs.size() != static_cast<size_t>(hd.n_xcorr) ||
            hd.gf.size() != fm::checked_mul(static_cast<size_t>(hd.n_xcorr), 6, "XCorr GF"))
            throw std::runtime_error("DataCache: incomplete XCorr phase " + std::to_string(i));
        if (window_length < 0)
            window_length = hd.n_xcorr;
        else if (hd.n_xcorr != window_length)
            throw std::runtime_error("DataCache: P/S XCorr window lengths differ");
        fm::validate_finite(hd.obs.data(), hd.obs.size(), "XCorr observation");
        fm::validate_finite(hd.gf.data(), hd.gf.size(), "XCorr Green function");
    }

    // ── Allocate flat arrays ──────────────────────────────────────────────

    if (has_xcorr) {
        // Clamp to first non-empty phase's window half-width (windows are fixed-size).
        int eff_maxlag = maxlag_;
        for (int i = 0; i < n_ph; ++i) {
            if (host_data[i].n_xcorr > 0) {
                eff_maxlag = std::min(maxlag_, (host_data[i].n_xcorr - 1) / 2);
                break;
            }
        }
        const int cc_rows = 2 * eff_maxlag + 1;
        entry.xcorr.maxlag = eff_maxlag;
        entry.xcorr.cc_rows = cc_rows;
        const size_t cc_count = fm::checked_mul(
            fm::checked_mul(static_cast<size_t>(n_ph), static_cast<size_t>(cc_rows), "XCorr cc"), 6,
            "XCorr cc components");
        entry.xcorr.cc = new double[cc_count];
        const size_t syn_phase_count =
            fm::checked_mul(static_cast<size_t>(n_ph), 6, "XCorr synamp phase stride");
        if (syn_phase_count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("XCorr synamp phase stride exceeds INT_MAX");
        entry.xcorr.n_syn_phases = static_cast<int>(syn_phase_count);
        const size_t synamp_count =
            fm::checked_mul(fm::checked_mul(static_cast<size_t>(n_ph), 36, "XCorr synamp"),
                            static_cast<size_t>(cc_rows), "XCorr synamp lags");
        entry.xcorr.synamp = new double[synamp_count];
        entry.xcorr.n_phases = n_ph;
        entry.xcorr.obs_norm2 = new double[n_ph];
    }

    if (has_polarity) {
        entry.polarity.n_phases = n_ph;
        entry.polarity.pol_vec =
            new double[fm::checked_mul(static_cast<size_t>(n_ph), 6, "polarity cache")];
        entry.polarity.obs_pol = new double[n_ph];
    }

    if (has_psr) {
        entry.psr.n_phases = n_ph;
        const size_t psr_count = fm::checked_mul(
            fm::checked_mul(static_cast<size_t>(n_ph), 6, "PSR cache"), 6, "PSR cache components");
        entry.psr.amp_P = new double[psr_count];
        entry.psr.amp_S = new double[psr_count];
        entry.psr.obs_psr = new double[n_ph];
    }

    // ── Compute reductions per phase (directly into flat arrays) ──────────

    // XCorr — per phase: compute CC, synamp, obs_norm2
    if (has_xcorr) {
        double *cc_total = entry.xcorr.cc;
        double *synamp_tot = entry.xcorr.synamp;
        double *obs_norm2 = entry.xcorr.obs_norm2;
        // Use the window-clamped stride from allocation; the raw maxlag_ may exceed
        // half the window and would index past the smaller array.
        const int cc_rows = entry.xcorr.cc_rows;
        const size_t syn_stride =
            fm::checked_mul(static_cast<size_t>(n_ph), 36, "XCorr synamp stride");
        const size_t cc_stride = fm::checked_mul(static_cast<size_t>(n_ph),
                                                 static_cast<size_t>(cc_rows), "XCorr cc stride");

        for (int i = 0; i < n_ph; ++i) {
            auto &hd = host_data[i];

            // obs_norm2 = sum(obs^2)
            double norm2 = 0.0;
            for (int j = 0; j < hd.n_xcorr; ++j) {
                norm2 += hd.obs[j] * hd.obs[j];
            }
            obs_norm2[i] = norm2;

            // per-lag synamp[6][6] = gf^T * gf (window shifted by `lag`)
            // stored column-major [n_ph × 36 × cc_rows]:
            //   synamp_tot[i + (a*6+b)*n_ph + lag_idx*(n_ph*36)]
            int N = hd.n_xcorr;
            // Half-width clamped to the window so a full window fits at lag 0.
            const int maxlag = std::min(maxlag_, (N - 1) / 2);
            entry.xcorr.maxlag = maxlag;
            for (int lag = -maxlag; lag <= maxlag; ++lag) {
                int lag_idx = lag + maxlag;
                for (int a = 0; a < 6; ++a) {
                    for (int b = a; b < 6; ++b) {
                        double sum = 0.0;
                        for (int t = 0; t < N; ++t) {
                            int t_shift = t - lag; // GF shifted by -lag (same as CC below)
                            if (t_shift >= 0 && t_shift < N) {
                                sum += hd.gf[t_shift * 6 + a] * hd.gf[t_shift * 6 + b];
                            }
                        }
                        const size_t ab_offset = static_cast<size_t>(i) +
                                                 static_cast<size_t>(a * 6 + b) * n_ph +
                                                 static_cast<size_t>(lag_idx) * syn_stride;
                        const size_t ba_offset = static_cast<size_t>(i) +
                                                 static_cast<size_t>(b * 6 + a) * n_ph +
                                                 static_cast<size_t>(lag_idx) * syn_stride;
                        synamp_tot[ab_offset] = sum;
                        synamp_tot[ba_offset] = sum; // symmetric
                    }
                }
            }

            // CC[2*maxlag+1][6] — time-domain cross-correlation
            // stored as [n_ph*cc_rows × 6] column-major
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
                    const size_t offset = static_cast<size_t>(i) * cc_rows + lag_idx +
                                          static_cast<size_t>(comp) * cc_stride;
                    cc_total[offset] = sum;
                }
            }
        }
    }

    // Polarity — per phase: sum gf_pol over time → pol_vec[6]
    if (has_polarity) {
        double *pol_vec_flat = entry.polarity.pol_vec;
        double *obs_pol_flat = entry.polarity.obs_pol;

        for (int i = 0; i < n_ph; ++i) {
            auto &hd = host_data[i];
            if (hd.n_pol == 0) {
                for (int c = 0; c < 6; ++c)
                    pol_vec_flat[i + c * n_ph] = 0.0;
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
        double *ampP_f = entry.psr.amp_P;
        double *ampS_f = entry.psr.amp_S;
        double *obs_f = entry.psr.obs_psr;

        for (int i = 0; i < n_ph; ++i) {
            auto &hd = host_data[i];
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

// ─ get_or_compute ─

const CacheEntry *DataCache::get_or_compute(int freq_idx, int depth_idx, int duration_idx) {
    CacheKey key {freq_idx, depth_idx, duration_idx};
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return &it->second;
    }
    throw std::runtime_error("DataCache: combo (" + std::to_string(freq_idx) + ", " +
                             std::to_string(depth_idx) + ", " + std::to_string(duration_idx) +
                             ") not loaded");
}

// ─ release_all ─

void DataCache::release_all() {
    for (auto &kv : cache_) {
        kv.second.release();
    }
    cache_.clear();
}

// ─ Static reduction helpers (stubs — reductions inline in load_combo) ─

void DataCache::compute_xcorr_reduction(CacheEntry & /*entry*/, const std::vector<double> & /*obs*/,
                                        const std::vector<double> & /*gf*/, int /*n_samples*/) {
    // Reductions are performed inline in load_combo().
}

void DataCache::compute_polarity_reduction(CacheEntry & /*entry*/,
                                           const std::vector<double> & /*gf_pol*/,
                                           int /*n_pol_samples*/) {
    // See load_combo() — inline host-side reduction.
}

void DataCache::compute_psr_reduction(CacheEntry & /*entry*/,
                                      const std::vector<double> & /*ampP_host*/,
                                      const std::vector<double> & /*ampS_host*/,
                                      const std::vector<double> & /*obs_psr_host*/) {
    // See load_combo() — PSR data is precomputed in database.h5, copied directly.
}
