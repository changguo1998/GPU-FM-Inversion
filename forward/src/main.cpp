#include <cmath>
#include <cstring>
#include <hdf5.h>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "backends/device.h"
#include "data_cache.h"
#include "hdf5_io.h"
#include "kernels/polarity_kernel.h"
#include "kernels/xcorr_kernel.h"
#include "mt_utils.h"

// ──────────────────────────────────────────────────────────────────────────
// main - forward stage entry point
//
// Usage: forward <database.h5> <status_N.h5>
//
// Reads preprocessed data + trials, runs misfit kernels, writes RAW
// INTERMEDIATE PRODUCTS to status_N.h5:/intermediates/{Module}/.
// Final misfit values (extract/compose) are produced by Julia assess.jl.
// ──────────────────────────────────────────────────────────────────────────

// Per-module config metadata read from database.h5:/config/{Module}/
struct ModuleConfig {
    std::string name;
    std::string op;      // "Xcorr" / "Polarity"
    std::string phase;   // "P" / "S"
    std::string channel; // "" or "H"/"V"
    bool is_composed;
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: forward <database.h5> <status_N.h5>" << std::endl;
        return 1;
    }

    std::string database_path = argv[1];
    std::string status_path = argv[2];

    try {
        // ══════════════════════════════════════════════════════════════
        // 1. Read trials from status_N.h5
        // ══════════════════════════════════════════════════════════════
        Hdf5Handle status_file;
        status_file.open(status_path.c_str(), H5F_ACC_RDWR);

        int N_trials = status_file.read_int_scalar("/trials/N_trials");

        auto strike_host = status_file.read_double_1d("/trials/strike");
        auto dip_host = status_file.read_double_1d("/trials/dip");
        auto rake_host = status_file.read_double_1d("/trials/rake");
        auto depth_host = status_file.read_double_1d("/trials/depth");
        auto d_idx_host = status_file.read_int_1d("/trials/depth_idx");
        auto f_idx_host = status_file.read_int_1d("/trials/freq_idx");

        std::vector<Trial> trials(N_trials);
        for (int i = 0; i < N_trials; ++i) {
            trials[i] = Trial {strike_host[i],
                               dip_host[i],
                               rake_host[i],
                               depth_host[i],
                               static_cast<int32_t>(d_idx_host[i]),
                               static_cast<int32_t>(f_idx_host[i])};
        }

        // ══════════════════════════════════════════════════════════════
        // 2. SDR -> MT conversion (host-side, degrees to radians)
        // ══════════════════════════════════════════════════════════════
        // XCorr uses [6 × N_trials] (LayoutLeft), Polarity uses [N_trials × 6]
        std::vector<double> mt_xcorr_host(static_cast<size_t>(6 * N_trials));
        std::vector<double> mt_pol_host(static_cast<size_t>(N_trials * 6));

        const double deg2rad = M_PI / 180.0;
        for (int t = 0; t < N_trials; ++t) {
            MomentTensor mt = sdr_to_mt(trials[t].strike * deg2rad, trials[t].dip * deg2rad,
                                        trials[t].rake * deg2rad);
            double comps[6] = {mt.Mxx, mt.Myy, mt.Mzz, mt.Mxy, mt.Mxz, mt.Myz};
            for (int c = 0; c < 6; ++c)
                mt_xcorr_host[c + t * 6] = comps[c];
            for (int c = 0; c < 6; ++c)
                mt_pol_host[t + c * N_trials] = comps[c];
        }

        // ══════════════════════════════════════════════════════════════
        // 3. Read module config from database.h5:/config
        // ══════════════════════════════════════════════════════════════
        Hdf5Handle db_reader;
        db_reader.open(database_path.c_str(), H5F_ACC_RDONLY);

        auto module_names = db_reader.read_string_1d("/config/misfit_modules");
        std::vector<ModuleConfig> modules;
        for (const auto &m : module_names) {
            ModuleConfig mc;
            mc.name = m;
            std::string base = "/config/" + m + "/";
            mc.op = db_reader.read_string_scalar((base + "operator").c_str());
            mc.is_composed = db_reader.read_int_scalar((base + "is_composed").c_str()) != 0;
            if (!mc.is_composed) {
                mc.phase = db_reader.read_string_scalar((base + "phase").c_str());
                mc.channel = db_reader.read_string_scalar((base + "channel").c_str());
            }
            modules.push_back(mc);
        }

        // Read P/S station indices (data partitioned by XcorrP/XcorrS groups)
        int n_p = 0, n_s = 0;
        std::vector<int> st_idx_vec;
        if (db_reader.group_exists("/XcorrP/station_idx")) {
            auto p_si = db_reader.read_int_1d("/XcorrP/station_idx");
            n_p = static_cast<int>(p_si.size());
            st_idx_vec.insert(st_idx_vec.end(), p_si.begin(), p_si.end());
        }
        if (db_reader.group_exists("/XcorrS/station_idx")) {
            auto s_si = db_reader.read_int_1d("/XcorrS/station_idx");
            n_s = static_cast<int>(s_si.size());
            st_idx_vec.insert(st_idx_vec.end(), s_si.begin(), s_si.end());
        }
        int N_phases = n_p + n_s;

        int N_stations = 0;
        for (int s : st_idx_vec)
            if (s + 1 > N_stations)
                N_stations = s + 1;

        // P-phase -> station, S-phase -> station maps (for polarity)
        std::vector<int> p_phase_of_station(N_stations, -1);
        std::vector<int> s_phase_of_station(N_stations, -1);
        for (int ph = 0; ph < n_p; ++ph) {
            int s = st_idx_vec[ph];
            if (s >= 0 && s < N_stations && p_phase_of_station[s] < 0)
                p_phase_of_station[s] = ph;
        }
        for (int ph = 0; ph < n_s; ++ph) {
            int s = st_idx_vec[n_p + ph];
            if (s >= 0 && s < N_stations && s_phase_of_station[s] < 0)
                s_phase_of_station[s] = n_p + ph;
        }

        db_reader.close();

        // ══════════════════════════════════════════════════════════════
        // 4. Initialize DataCache, load preprocessed data
        // ══════════════════════════════════════════════════════════════
        const int maxlag = 50;
        const int cc_pp = 2 * maxlag + 1;

        DataCache cache(maxlag);
        cache.load_from_database(database_path, trials);

        // Collect unique (freq_idx, depth_idx) combos from trials
        std::set<std::pair<int, int>> combo_set;
        for (const auto &t : trials)
            combo_set.insert({t.freq_idx, t.depth_idx});
        std::vector<std::pair<int, int>> combos(combo_set.begin(), combo_set.end());

        // ══════════════════════════════════════════════════════════════
        // 5. Allocate intermediate output arrays (accumulated across combos)
        // ══════════════════════════════════════════════════════════════
        bool has_xcorr_p = n_p > 0;
        bool has_xcorr_s = n_s > 0;
        bool has_polarity = N_stations > 0;

        std::vector<double> cc_max_p(has_xcorr_p ? (size_t)n_p * N_trials : 0, 0.0);
        std::vector<int32_t> best_lag_p(has_xcorr_p ? (size_t)n_p * N_trials : 0, 0);
        std::vector<double> cc_max_s(has_xcorr_s ? (size_t)n_s * N_trials : 0, 0.0);
        std::vector<int32_t> best_lag_s(has_xcorr_s ? (size_t)n_s * N_trials : 0, 0);
        std::vector<int8_t> syn_sign(has_polarity ? (size_t)N_stations * N_trials : 0, 0);
        std::vector<double> dot_value(has_polarity ? (size_t)N_stations * N_trials : 0,
                                      std::numeric_limits<double>::quiet_NaN());

        // ══════════════════════════════════════════════════════════════
        // 6. Launch kernels per combo, accumulate into intermediate arrays
        // ══════════════════════════════════════════════════════════════
        for (const auto &combo : combos) {
            int f_idx = combo.first;
            int d_idx = combo.second;

            std::vector<int> trial_indices;
            for (int t = 0; t < N_trials; ++t)
                if (trials[t].freq_idx == f_idx && trials[t].depth_idx == d_idx)
                    trial_indices.push_back(t);
            if (trial_indices.empty())
                continue;

            int n_sub = static_cast<int>(trial_indices.size());
            const CacheEntry *entry = nullptr;
            try {
                entry = cache.get_or_compute(f_idx, d_idx);
            } catch (const std::runtime_error &) {
                continue;
            }
            if (!entry || !entry->valid())
                continue;

            // ── Build MT sub-views for this combo's trials ──
            std::vector<double> mt_xcorr_sub(static_cast<size_t>(6 * n_sub));
            std::vector<double> mt_pol_sub(static_cast<size_t>(n_sub * 6));
            for (int si = 0; si < n_sub; ++si) {
                int ti = trial_indices[si];
                for (int c = 0; c < 6; ++c) {
                    mt_xcorr_sub[c + si * 6] = mt_xcorr_host[c + ti * 6];
                    mt_pol_sub[si + c * n_sub] = mt_pol_host[ti + c * N_trials];
                }
            }

            // ── XCorr: cc_max + best_lag per (phase, trial) ──
            if (entry->xcorr.cc != nullptr && N_phases > 0) {
                std::vector<double> synamp_r(static_cast<size_t>(N_phases * 36));
                const double *synamp_src = entry->xcorr.synamp;
                int n_syn_phases = entry->xcorr.n_syn_phases;
                for (int p = 0; p < N_phases; ++p)
                    for (int i = 0; i < 6; ++i)
                        for (int j = 0; j < 6; ++j)
                            synamp_r[p + (i * 6 + j) * N_phases] =
                                synamp_src[(p * 6 + i) + j * n_syn_phases];

                std::vector<double> cc_max_sub(static_cast<size_t>(N_phases * n_sub));
                std::vector<int32_t> best_lag_sub(static_cast<size_t>(N_phases * n_sub));

                fm::launch_xcorr_misfit<Backend::OpenMP>(
                    mt_xcorr_sub.data(), entry->xcorr.cc, synamp_r.data(), entry->xcorr.obs_norm2,
                    cc_max_sub.data(), best_lag_sub.data(), N_phases, n_sub, cc_pp, maxlag);

                // Write back: split P (rows 0..n_p) and S (rows n_p..N_phases)
                for (int ph = 0; ph < N_phases; ++ph)
                    for (int si = 0; si < n_sub; ++si) {
                        double v = cc_max_sub[ph + si * N_phases];
                        int32_t lag = best_lag_sub[ph + si * N_phases];
                        if (ph < n_p) {
                            cc_max_p[ph * N_trials + trial_indices[si]] = v;
                            best_lag_p[ph * N_trials + trial_indices[si]] = lag;
                        } else {
                            int sp = ph - n_p;
                            cc_max_s[sp * N_trials + trial_indices[si]] = v;
                            best_lag_s[sp * N_trials + trial_indices[si]] = lag;
                        }
                    }
            }

            // ── Polarity: syn_sign + dot_value per (station, trial) ──
            if (entry->polarity.pol_vec != nullptr && N_stations > 0) {
                std::vector<double> pol_vec_s(static_cast<size_t>(N_stations * 6));
                const double *pol_src = entry->polarity.pol_vec;
                int n_phases_pol = entry->polarity.n_phases;

                for (int s = 0; s < N_stations; ++s) {
                    int pp = p_phase_of_station[s];
                    if (pp >= 0) {
                        for (int c = 0; c < 6; ++c)
                            pol_vec_s[s + c * N_stations] = pol_src[pp + c * n_phases_pol];
                    } else {
                        for (int c = 0; c < 6; ++c)
                            pol_vec_s[s + c * N_stations] = 0.0;
                    }
                }

                std::vector<int8_t> syn_sign_sub(static_cast<size_t>(N_stations * n_sub));
                std::vector<double> dot_sub(static_cast<size_t>(N_stations * n_sub));

                fm::launch_polarity_kernel<Backend::OpenMP>(mt_pol_sub.data(), pol_vec_s.data(),
                                                            syn_sign_sub.data(), dot_sub.data(),
                                                            N_stations, n_sub);

                for (int s = 0; s < N_stations; ++s)
                    for (int si = 0; si < n_sub; ++si) {
                        syn_sign[s * N_trials + trial_indices[si]] =
                            syn_sign_sub[s + si * N_stations];
                        dot_value[s * N_trials + trial_indices[si]] = dot_sub[s + si * N_stations];
                    }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // 7. Write intermediates to status_N.h5:/intermediates/
        // ══════════════════════════════════════════════════════════════
        if (!status_file.group_exists("/intermediates"))
            status_file.create_group("/intermediates");

        auto write_xcorr_inter = [&](const char *key, const std::vector<double> &cc,
                                     const std::vector<int32_t> &lag, int n_ph) {
            if (n_ph == 0)
                return;
            std::string g = std::string("/intermediates/") + key;
            if (!status_file.group_exists(g.c_str()))
                status_file.create_group(g.c_str());
            status_file.write_double_2d((g + "/cc_max").c_str(), cc.data(), (hsize_t)n_ph,
                                        (hsize_t)N_trials);
            status_file.write_int32_2d((g + "/best_lag").c_str(), lag.data(), (hsize_t)n_ph,
                                       (hsize_t)N_trials);
        };

        // Map module -> canonical intermediate key (operator + phase [+ channel]).
        // Dedup: instances sharing a key (e.g. XcorrP + AbsShiftP) write once.
        std::set<std::string> written_keys;
        for (const auto &mc : modules) {
            if (mc.is_composed)
                continue; // composed misfits have no intermediates (Julia-only)
            std::string key = mc.op + mc.phase;
            if (!mc.channel.empty())
                key += "_" + mc.channel;
            if (!written_keys.insert(key).second)
                continue; // already written for this canonical key
            if (mc.op == "Xcorr" && mc.phase == "P")
                write_xcorr_inter(key.c_str(), cc_max_p, best_lag_p, n_p);
            else if (mc.op == "Xcorr" && mc.phase == "S")
                write_xcorr_inter(key.c_str(), cc_max_s, best_lag_s, n_s);
            else if (mc.op == "Polarity") {
                std::string g = "/intermediates/" + key;
                if (!status_file.group_exists(g.c_str()))
                    status_file.create_group(g.c_str());
                status_file.write_int8_2d((g + "/syn_sign").c_str(), syn_sign.data(),
                                          (hsize_t)N_stations, (hsize_t)N_trials);
                status_file.write_double_2d((g + "/dot_value").c_str(), dot_value.data(),
                                            (hsize_t)N_stations, (hsize_t)N_trials);
            }
        }

        status_file.close();
        cache.release_all();

        std::cout << "fm_forward: " << N_trials << " trials × " << combos.size() << " combos -> "
                  << N_phases << " phases, " << N_stations << " stations"
                  << " (intermediates written)" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "fm_forward error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
