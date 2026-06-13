#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <limits>
#include <cstring>
#include <cmath>

#include "hdf5_io.h"
#include "mt_utils.h"
#include "data_cache.h"
#include "kernels/xcorr_kernel.h"
#include "kernels/polarity_kernel.h"
#include "kernels/psr_kernel.h"

// ──────────────────────────────────────────────────────────────────────────
// Helper: read HDF5 variable-length string 1D dataset
// ──────────────────────────────────────────────────────────────────────────
static std::vector<std::string> read_string_1d(hid_t file_id, const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open " + std::string(path));

    hid_t space = H5Dget_space(dset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

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
// main — forward stage entry point
//
// Usage: forward <database.h5> <status_N.h5>
//
// No weights. No aggregation. No strategy knowledge.
// ──────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: forward <database.h5> <status_N.h5>" << std::endl;
        return 1;
    }

    std::string database_path = argv[1];
    std::string status_path   = argv[2];

    Kokkos::initialize(argc, argv);
    {
        try {
            auto exec = Kokkos::DefaultExecutionSpace();

            // ══════════════════════════════════════════════════════════════
            // 1. Read trials from status_N.h5
            // ══════════════════════════════════════════════════════════════
            Hdf5Handle status_file;
            status_file.open(status_path.c_str(), H5F_ACC_RDWR);

            int N_trials = status_file.read_int_scalar("/trials/N_trials");

            auto strike_host = status_file.read_double_1d("/trials/strike");
            auto dip_host    = status_file.read_double_1d("/trials/dip");
            auto rake_host   = status_file.read_double_1d("/trials/rake");
            auto depth_host  = status_file.read_double_1d("/trials/depth");
            auto d_idx_host  = status_file.read_int_1d("/trials/depth_idx");
            auto f_idx_host  = status_file.read_int_1d("/trials/freq_idx");

            std::vector<Trial> trials(N_trials);
            for (int i = 0; i < N_trials; ++i) {
                trials[i] = Trial{
                    strike_host[i], dip_host[i], rake_host[i], depth_host[i],
                    static_cast<int32_t>(d_idx_host[i]),
                    static_cast<int32_t>(f_idx_host[i])
                };
            }

            // ══════════════════════════════════════════════════════════════
            // 2. SDR → MT conversion (host-side, degrees to radians)
            // ══════════════════════════════════════════════════════════════
            // Two layouts: XCorr uses [6 × N_trials], Polarity/PSR use [N_trials × 6]
            Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
                mt_xcorr_host("mt_xcorr_h", 6, N_trials);
            Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
                mt_pol_host("mt_pol_h", N_trials, 6);

            const double deg2rad = M_PI / 180.0;
            for (int t = 0; t < N_trials; ++t) {
                MomentTensor mt = sdr_to_mt(
                    trials[t].strike * deg2rad,
                    trials[t].dip    * deg2rad,
                    trials[t].rake   * deg2rad
                );
                double comps[6] = {mt.Mxx, mt.Myy, mt.Mzz, mt.Mxy, mt.Mxz, mt.Myz};

                // LayoutLeft [6, N]: element (row, col) at row + col*6
                for (int c = 0; c < 6; ++c)
                    mt_xcorr_host(c, t) = comps[c];

                // LayoutLeft [N, 6]: element (row, col) at row + col*N
                for (int c = 0; c < 6; ++c)
                    mt_pol_host(t, c) = comps[c];
            }

            // Copy MT to device
            auto mt_xcorr_dev = Kokkos::create_mirror_view_and_copy(
                Kokkos::DefaultExecutionSpace(), mt_xcorr_host);
            auto mt_pol_dev = Kokkos::create_mirror_view_and_copy(
                Kokkos::DefaultExecutionSpace(), mt_pol_host);

            // ══════════════════════════════════════════════════════════════
            // 3. Read station/phase index from database.h5
            // ══════════════════════════════════════════════════════════════
            Hdf5Handle db_reader;
            db_reader.open(database_path.c_str(), H5F_ACC_RDONLY);

            // Read phase types and station mapping
            hid_t db_raw = db_reader.file_id;
            auto phase_type = read_string_1d(db_raw, "/index/phase_type");
            auto st_idx_vec = db_reader.read_int_1d("/index/station_idx");

            int N_phases   = static_cast<int>(phase_type.size());
            int N_stations = 0;
            for (int s : st_idx_vec)
                if (s + 1 > N_stations) N_stations = s + 1;

            // Build station → (P_phase_idx, S_phase_idx) map
            std::vector<int> p_phase_of_station(N_stations, -1);
            std::vector<int> s_phase_of_station(N_stations, -1);
            for (int ph = 0; ph < N_phases; ++ph) {
                int s = st_idx_vec[ph];
                if (phase_type[ph] == "P" && p_phase_of_station[s] < 0)
                    p_phase_of_station[s] = ph;
                if (phase_type[ph] == "S" && s_phase_of_station[s] < 0)
                    s_phase_of_station[s] = ph;
            }

            db_reader.close();

            // ══════════════════════════════════════════════════════════════
            // 4. Initialize DataCache, load preprocessed data
            // ══════════════════════════════════════════════════════════════
            // Sensible default for XCorr maxlag; production should read
            // from database config.
            const int maxlag = 50;
            const int cc_pp  = 2 * maxlag + 1;

            DataCache cache(maxlag);
            cache.load_from_database(database_path, trials);

            // Collect unique (freq_idx, depth_idx) combos from trials
            std::set<std::pair<int,int>> combo_set;
            for (const auto& t : trials)
                combo_set.insert({t.freq_idx, t.depth_idx});
            std::vector<std::pair<int,int>> combos(combo_set.begin(), combo_set.end());

            // ══════════════════════════════════════════════════════════════
            // 5. Allocate host output arrays
            // ══════════════════════════════════════════════════════════════
            std::vector<double> xcorr_out(N_phases * N_trials,
                                          std::numeric_limits<double>::quiet_NaN());
            std::vector<double> polarity_out(N_stations * N_trials,
                                              std::numeric_limits<double>::quiet_NaN());
            std::vector<double> psr_out(N_stations * N_trials,
                                         std::numeric_limits<double>::quiet_NaN());

            // ══════════════════════════════════════════════════════════════
            // 6. Launch XCorr + Polarity + PSR kernels per combo
            // ══════════════════════════════════════════════════════════════
            for (const auto& combo : combos) {
                int f_idx = combo.first;
                int d_idx = combo.second;

                // Find trial indices for this combo
                std::vector<int> trial_indices;
                for (int t = 0; t < N_trials; ++t)
                    if (trials[t].freq_idx == f_idx && trials[t].depth_idx == d_idx)
                        trial_indices.push_back(t);

                if (trial_indices.empty()) continue;

                int n_sub = static_cast<int>(trial_indices.size());
                const CacheEntry* entry = nullptr;
                try {
                    entry = cache.get_or_compute(f_idx, d_idx);
                } catch (const std::runtime_error&) {
                    // Data not available for this combo — skip
                    continue;
                }
                if (!entry || !entry->valid()) continue;

                // ── Build MT sub-views for this combo's trials ────────────
                Kokkos::View<double**, Kokkos::LayoutLeft,
                             Kokkos::DefaultExecutionSpace>
                    mt_xcorr_sub("mt_xcorr_sub", 6, n_sub);
                Kokkos::View<double**, Kokkos::LayoutLeft,
                             Kokkos::DefaultExecutionSpace>
                    mt_pol_sub("mt_pol_sub", n_sub, 6);

                auto mt_xcorr_sub_h = Kokkos::create_mirror_view(mt_xcorr_sub);
                auto mt_pol_sub_h   = Kokkos::create_mirror_view(mt_pol_sub);
                for (int si = 0; si < n_sub; ++si) {
                    int ti = trial_indices[si];
                    for (int c = 0; c < 6; ++c) {
                        mt_xcorr_sub_h(c, si) = mt_xcorr_host(c, ti);
                        mt_pol_sub_h(si, c)   = mt_pol_host(ti, c);
                    }
                }
                Kokkos::deep_copy(exec, mt_xcorr_sub, mt_xcorr_sub_h);
                Kokkos::deep_copy(exec, mt_pol_sub,   mt_pol_sub_h);

                // ── XCorr ─────────────────────────────────────────────────
                if (entry->xcorr.cc.span() > 0) {
                    // Reshape synamp: cache stores [N_ph*6, 6]; kernel expects [N_ph, 36]
                    Kokkos::View<double**, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        synamp_r("synamp_reshaped", N_phases, 36);

                    auto synamp_src_h = Kokkos::create_mirror_view(entry->xcorr.synamp);
                    Kokkos::deep_copy(synamp_src_h, entry->xcorr.synamp);

                    auto synamp_dst_h = Kokkos::create_mirror_view(synamp_r);
                    for (int p = 0; p < N_phases; ++p)
                        for (int i = 0; i < 6; ++i)
                            for (int j = 0; j < 6; ++j)
                                synamp_dst_h(p, i * 6 + j) = synamp_src_h(p * 6 + i, j);
                    Kokkos::deep_copy(exec, synamp_r, synamp_dst_h);

                    // Output: [N_phases × n_sub]
                    Kokkos::View<double**, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        xcorr_sub("xcorr_sub", N_phases, n_sub);

                    fm::launch_xcorr_misfit(
                        exec,
                        mt_xcorr_sub,                 // [6 × n_sub] LayoutLeft
                        entry->xcorr.cc,              // [N_ph·cc_pp × 6] LayoutLeft
                        synamp_r,                     // [N_ph × 36] LayoutLeft
                        entry->xcorr.obs_norm2,       // [N_ph]
                        xcorr_sub,                    // [N_ph × n_sub] LayoutLeft
                        N_phases, n_sub, cc_pp);

                    auto xcorr_sub_h = Kokkos::create_mirror_view(xcorr_sub);
                    Kokkos::deep_copy(xcorr_sub_h, xcorr_sub);
                    for (int ph = 0; ph < N_phases; ++ph)
                        for (int si = 0; si < n_sub; ++si)
                            xcorr_out[ph * N_trials + trial_indices[si]] =
                                xcorr_sub_h(ph, si);
                }

                // ── Polarity ──────────────────────────────────────────────
                if (entry->polarity.pol_vec.span() > 0) {
                    // Map per-phase → per-station:
                    // Cache pol_vec is [N_ph·6]; kernel expects [N_st × 6]
                    Kokkos::View<double**, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        pol_vec_s("pol_vec_station", N_stations, 6);
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace>
                        obs_pol_s("obs_pol_station", N_stations);

                    auto pol_src_h = Kokkos::create_mirror_view(entry->polarity.pol_vec);
                    auto obs_src_h = Kokkos::create_mirror_view(entry->polarity.obs_pol);
                    Kokkos::deep_copy(pol_src_h, entry->polarity.pol_vec);
                    Kokkos::deep_copy(obs_src_h, entry->polarity.obs_pol);

                    auto pol_dst_h = Kokkos::create_mirror_view(pol_vec_s);
                    auto obs_dst_h = Kokkos::create_mirror_view(obs_pol_s);

                    for (int s = 0; s < N_stations; ++s) {
                        int pp = p_phase_of_station[s];
                        if (pp >= 0) {
                            for (int c = 0; c < 6; ++c)
                                pol_dst_h(s, c) = pol_src_h(pp * 6 + c);
                            obs_dst_h(s) = obs_src_h(pp);
                        } else {
                            for (int c = 0; c < 6; ++c)
                                pol_dst_h(s, c) = 0.0;
                            obs_dst_h(s) = std::numeric_limits<double>::quiet_NaN();
                        }
                    }

                    Kokkos::deep_copy(exec, pol_vec_s, pol_dst_h);
                    Kokkos::deep_copy(exec, obs_pol_s, obs_dst_h);

                    Kokkos::View<double**, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        pol_sub("polarity_sub", N_stations, n_sub);

                    fm::launch_polarity_kernel(
                        mt_pol_sub,    // [n_sub × 6] LayoutLeft
                        pol_vec_s,     // [N_st × 6] LayoutLeft
                        obs_pol_s,     // [N_st]
                        pol_sub);      // [N_st × n_sub] LayoutLeft

                    auto pol_sub_h = Kokkos::create_mirror_view(pol_sub);
                    Kokkos::deep_copy(pol_sub_h, pol_sub);
                    for (int s = 0; s < N_stations; ++s)
                        for (int si = 0; si < n_sub; ++si)
                            polarity_out[s * N_trials + trial_indices[si]] =
                                pol_sub_h(s, si);
                }

                // ── PSR ───────────────────────────────────────────────────
                if (entry->psr.amp_P.span() > 0) {
                    // Map per-phase → per-station:
                    // Cache amp_P/amp_S are [N_ph·6, 6]; kernel expects [N_st × 6 × 6]
                    Kokkos::View<double***, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        ampP_s("ampP_station", N_stations, 6, 6);
                    Kokkos::View<double***, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        ampS_s("ampS_station", N_stations, 6, 6);
                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace>
                        obs_psr_s("obs_psr_station", N_stations);

                    auto ampP_src_h = Kokkos::create_mirror_view(entry->psr.amp_P);
                    auto ampS_src_h = Kokkos::create_mirror_view(entry->psr.amp_S);
                    auto opsr_src_h = Kokkos::create_mirror_view(entry->psr.obs_psr);
                    Kokkos::deep_copy(ampP_src_h, entry->psr.amp_P);
                    Kokkos::deep_copy(ampS_src_h, entry->psr.amp_S);
                    Kokkos::deep_copy(opsr_src_h, entry->psr.obs_psr);

                    auto ampP_dst_h = Kokkos::create_mirror_view(ampP_s);
                    auto ampS_dst_h = Kokkos::create_mirror_view(ampS_s);
                    auto opsr_dst_h = Kokkos::create_mirror_view(obs_psr_s);

                    for (int s = 0; s < N_stations; ++s) {
                        int pp = p_phase_of_station[s];
                        int sp = s_phase_of_station[s];
                        if (pp >= 0 && sp >= 0) {
                            for (int i = 0; i < 6; ++i)
                                for (int j = 0; j < 6; ++j) {
                                    ampP_dst_h(s, i, j) = ampP_src_h(pp * 6 + i, j);
                                    ampS_dst_h(s, i, j) = ampS_src_h(sp * 6 + i, j);
                                }
                            opsr_dst_h(s) = opsr_src_h(pp);
                        } else {
                            for (int i = 0; i < 6; ++i)
                                for (int j = 0; j < 6; ++j) {
                                    ampP_dst_h(s, i, j) = 0.0;
                                    ampS_dst_h(s, i, j) = 0.0;
                                }
                            opsr_dst_h(s) = std::numeric_limits<double>::quiet_NaN();
                        }
                    }

                    Kokkos::deep_copy(exec, ampP_s,    ampP_dst_h);
                    Kokkos::deep_copy(exec, ampS_s,    ampS_dst_h);
                    Kokkos::deep_copy(exec, obs_psr_s, opsr_dst_h);

                    Kokkos::View<double**, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>
                        psr_sub("psr_sub", N_stations, n_sub);

                    fm::launch_psr_kernel(
                        mt_pol_sub,   // [n_sub × 6] LayoutLeft
                        ampP_s,       // [N_st × 6 × 6] LayoutLeft
                        ampS_s,       // [N_st × 6 × 6] LayoutLeft
                        obs_psr_s,    // [N_st]
                        psr_sub);     // [N_st × n_sub] LayoutLeft

                    auto psr_sub_h = Kokkos::create_mirror_view(psr_sub);
                    Kokkos::deep_copy(psr_sub_h, psr_sub);
                    for (int s = 0; s < N_stations; ++s)
                        for (int si = 0; si < n_sub; ++si)
                            psr_out[s * N_trials + trial_indices[si]] =
                                psr_sub_h(s, si);
                }
            }

            exec.fence();

            // ══════════════════════════════════════════════════════════════
            // 7. Write misfits to status_N.h5
            // ══════════════════════════════════════════════════════════════
            if (!status_file.group_exists("/misfits"))
                status_file.create_group("/misfits");

            status_file.write_double_2d("/misfits/xcorr",
                xcorr_out.data(),
                static_cast<hsize_t>(N_phases),
                static_cast<hsize_t>(N_trials));

            status_file.write_double_2d("/misfits/polarity",
                polarity_out.data(),
                static_cast<hsize_t>(N_stations),
                static_cast<hsize_t>(N_trials));

            status_file.write_double_2d("/misfits/psr",
                psr_out.data(),
                static_cast<hsize_t>(N_stations),
                static_cast<hsize_t>(N_trials));

            status_file.close();

            // Free GPU memory
            cache.release_all();

            std::cout << "fm_forward: " << N_trials << " trials × "
                      << combos.size() << " combos → "
                      << N_phases << " phases, " << N_stations << " stations"
                      << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "fm_forward error: " << e.what() << std::endl;
            Kokkos::finalize();
            return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}