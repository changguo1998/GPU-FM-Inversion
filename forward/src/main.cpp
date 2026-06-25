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
#include "kernels/psr_kernel.h"
#include "kernels/xcorr_kernel.h"
#include "mt_utils.h"

// ──────────────────────────────────────────────────────────────────────────
// Helper: read HDF5 variable-length string 1D dataset
// ──────────────────────────────────────────────────────────────────────────
static std::vector<std::string> read_string_1d(hid_t file_id,
                                               const char *path) {
  hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
  if (dset < 0)
    throw std::runtime_error("Cannot open " + std::string(path));

  hid_t space = H5Dget_space(dset);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, nullptr);

  std::vector<char *> buf(dims[0]);
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
      trials[i] = Trial{strike_host[i],
                        dip_host[i],
                        rake_host[i],
                        depth_host[i],
                        static_cast<int32_t>(d_idx_host[i]),
                        static_cast<int32_t>(f_idx_host[i])};
    }

    // ══════════════════════════════════════════════════════════════
    // 2. SDR → MT conversion (host-side, degrees to radians)
    // ══════════════════════════════════════════════════════════════
    // Two layouts: XCorr uses [6 × N_trials], Polarity/PSR use [N_trials × 6]
    std::vector<double> mt_xcorr_host(static_cast<size_t>(6 * N_trials));
    std::vector<double> mt_pol_host(static_cast<size_t>(N_trials * 6));

    const double deg2rad = M_PI / 180.0;
    for (int t = 0; t < N_trials; ++t) {
      MomentTensor mt =
          sdr_to_mt(trials[t].strike * deg2rad, trials[t].dip * deg2rad,
                    trials[t].rake * deg2rad);
      double comps[6] = {mt.Mxx, mt.Myy, mt.Mzz, mt.Mxy, mt.Mxz, mt.Myz};

      // LayoutLeft [6, N]: element (row, col) at row + col*6
      for (int c = 0; c < 6; ++c)
        mt_xcorr_host[c + t * 6] = comps[c];

      // LayoutLeft [N, 6]: element (row, col) at row + col*N
      for (int c = 0; c < 6; ++c)
        mt_pol_host[t + c * N_trials] = comps[c];
    }

    // ══════════════════════════════════════════════════════════════
    // 3. Read station/phase index from database.h5
    // ══════════════════════════════════════════════════════════════
    Hdf5Handle db_reader;
    db_reader.open(database_path.c_str(), H5F_ACC_RDONLY);

    // Read phase types and station mapping
    hid_t db_raw = db_reader.file_id;
    auto phase_type = read_string_1d(db_raw, "/index/phase_type");
    auto st_idx_vec = db_reader.read_int_1d("/index/station_idx");

    int N_phases = static_cast<int>(phase_type.size());
    int N_stations = 0;
    for (int s : st_idx_vec)
      if (s + 1 > N_stations)
        N_stations = s + 1;

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
    const int cc_pp = 2 * maxlag + 1;

    DataCache cache(maxlag);
    cache.load_from_database(database_path, trials);

    // Collect unique (freq_idx, depth_idx) combos from trials
    std::set<std::pair<int, int>> combo_set;
    for (const auto &t : trials)
      combo_set.insert({t.freq_idx, t.depth_idx});
    std::vector<std::pair<int, int>> combos(combo_set.begin(), combo_set.end());

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
    for (const auto &combo : combos) {
      int f_idx = combo.first;
      int d_idx = combo.second;

      // Find trial indices for this combo
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
        // Data not available for this combo — skip
        continue;
      }
      if (!entry || !entry->valid())
        continue;

      // ── Build MT sub-views for this combo's trials ────────────
      std::vector<double> mt_xcorr_sub(static_cast<size_t>(6 * n_sub));
      std::vector<double> mt_pol_sub(static_cast<size_t>(n_sub * 6));
      for (int si = 0; si < n_sub; ++si) {
        int ti = trial_indices[si];
        for (int c = 0; c < 6; ++c) {
          mt_xcorr_sub[c + si * 6] = mt_xcorr_host[c + ti * 6];
          mt_pol_sub[si + c * n_sub] = mt_pol_host[ti + c * N_trials];
        }
      }

      // ── XCorr ─────────────────────────────────────────────────
      if (entry->xcorr.cc != nullptr) {
        // Reshape synamp: cache stores [N_ph*6, 6]; kernel expects [N_ph, 36]
        std::vector<double> synamp_r(static_cast<size_t>(N_phases * 36));
        const double *synamp_src = entry->xcorr.synamp;
        int n_syn_phases = entry->xcorr.n_syn_phases;
        for (int p = 0; p < N_phases; ++p)
          for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
              synamp_r[p + (i * 6 + j) * N_phases] =
                  synamp_src[(p * 6 + i) + j * n_syn_phases];

        // Output: [N_phases × n_sub]
        std::vector<double> xcorr_sub(static_cast<size_t>(N_phases * n_sub));

        fm::launch_xcorr_misfit<Backend::OpenMP>(
            mt_xcorr_sub.data(),    // [6 × n_sub] col-major
            entry->xcorr.cc,        // [N_ph·cc_pp × 6] col-major
            synamp_r.data(),        // [N_ph × 36] col-major
            entry->xcorr.obs_norm2, // [N_ph]
            xcorr_sub.data(),       // [N_ph × n_sub] col-major
            N_phases, n_sub, cc_pp);

        // Write back
        for (int ph = 0; ph < N_phases; ++ph)
          for (int si = 0; si < n_sub; ++si)
            xcorr_out[ph * N_trials + trial_indices[si]] =
                xcorr_sub[ph + si * N_phases];
      }

      // ── Polarity ──────────────────────────────────────────────
      if (entry->polarity.pol_vec != nullptr) {
        // Map per-phase → per-station:
        std::vector<double> pol_vec_s(static_cast<size_t>(N_stations * 6));
        std::vector<double> obs_pol_s(N_stations);

        const double *pol_src = entry->polarity.pol_vec;
        const double *obs_src = entry->polarity.obs_pol;
        int n_phases_pol = entry->polarity.n_phases;

        for (int s = 0; s < N_stations; ++s) {
          int pp = p_phase_of_station[s];
          if (pp >= 0) {
            for (int c = 0; c < 6; ++c)
              pol_vec_s[s + c * N_stations] = pol_src[pp + c * n_phases_pol];
            obs_pol_s[s] = obs_src[pp];
          } else {
            for (int c = 0; c < 6; ++c)
              pol_vec_s[s + c * N_stations] = 0.0;
            obs_pol_s[s] = std::numeric_limits<double>::quiet_NaN();
          }
        }

        std::vector<double> pol_sub(static_cast<size_t>(N_stations * n_sub));

        fm::launch_polarity_kernel<Backend::OpenMP>(
            mt_pol_sub.data(), // [n_sub × 6] col-major
            pol_vec_s.data(),  // [N_st × 6] col-major
            obs_pol_s.data(),  // [N_st]
            pol_sub.data(),    // [N_st × n_sub] col-major
            N_stations, n_sub);

        for (int s = 0; s < N_stations; ++s)
          for (int si = 0; si < n_sub; ++si)
            polarity_out[s * N_trials + trial_indices[si]] =
                pol_sub[s + si * N_stations];
      }

      // ── PSR ───────────────────────────────────────────────────
      if (entry->psr.amp_P != nullptr) {
        // Map per-phase → per-station:
        std::vector<double> ampP_s(static_cast<size_t>(N_stations * 6 * 6));
        std::vector<double> ampS_s(static_cast<size_t>(N_stations * 6 * 6));
        std::vector<double> obs_psr_s(N_stations);

        const double *ampP_src = entry->psr.amp_P;
        const double *ampS_src = entry->psr.amp_S;
        const double *opsr_src = entry->psr.obs_psr;
        int n_phases_psr = entry->psr.n_phases;

        for (int s = 0; s < N_stations; ++s) {
          int pp = p_phase_of_station[s];
          int sp = s_phase_of_station[s];
          if (pp >= 0 && sp >= 0) {
            for (int i = 0; i < 6; ++i)
              for (int j = 0; j < 6; ++j) {
                ampP_s[s + i * N_stations + j * (N_stations * 6)] =
                    ampP_src[pp + i * n_phases_psr + j * (n_phases_psr * 6)];
                ampS_s[s + i * N_stations + j * (N_stations * 6)] =
                    ampS_src[sp + i * n_phases_psr + j * (n_phases_psr * 6)];
              }
            obs_psr_s[s] = opsr_src[pp];
          } else {
            for (int i = 0; i < 6; ++i)
              for (int j = 0; j < 6; ++j) {
                ampP_s[s + i * N_stations + j * (N_stations * 6)] = 0.0;
                ampS_s[s + i * N_stations + j * (N_stations * 6)] = 0.0;
              }
            obs_psr_s[s] = std::numeric_limits<double>::quiet_NaN();
          }
        }

        std::vector<double> psr_sub(static_cast<size_t>(N_stations * n_sub));

        fm::launch_psr_kernel<Backend::OpenMP>(
            mt_pol_sub.data(), // [n_sub × 6] col-major
            ampP_s.data(),     // [N_st × 6 × 6] col-major
            ampS_s.data(),     // [N_st × 6 × 6] col-major
            obs_psr_s.data(),  // [N_st]
            psr_sub.data(),    // [N_st × n_sub] col-major
            N_stations, n_sub);

        for (int s = 0; s < N_stations; ++s)
          for (int si = 0; si < n_sub; ++si)
            psr_out[s * N_trials + trial_indices[si]] =
                psr_sub[s + si * N_stations];
      }
    }

    // ══════════════════════════════════════════════════════════════
    // 7. Write misfits to status_N.h5
    // ══════════════════════════════════════════════════════════════
    if (!status_file.group_exists("/misfits"))
      status_file.create_group("/misfits");

    status_file.write_double_2d("/misfits/xcorr", xcorr_out.data(),
                                static_cast<hsize_t>(N_phases),
                                static_cast<hsize_t>(N_trials));

    status_file.write_double_2d("/misfits/polarity", polarity_out.data(),
                                static_cast<hsize_t>(N_stations),
                                static_cast<hsize_t>(N_trials));

    status_file.write_double_2d("/misfits/psr", psr_out.data(),
                                static_cast<hsize_t>(N_stations),
                                static_cast<hsize_t>(N_trials));

    status_file.close();

    // Free GPU memory
    cache.release_all();

    std::cout << "fm_forward: " << N_trials << " trials × " << combos.size()
              << " combos → " << N_phases << " phases, " << N_stations
              << " stations" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "fm_forward error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}