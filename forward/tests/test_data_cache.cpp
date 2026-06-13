#include "data_cache.h"
#include "hdf5_io.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>

// ──────────────────────────────────────────────────────────────────────────
// Test infrastructure
// ──────────────────────────────────────────────────────────────────────────

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << std::endl; \
        failures++; \
    } else { \
        std::cout << "  OK: " << msg << std::endl; \
    } \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL: " << msg << " expected=" << (b) << " got=" << (a) << std::endl; \
        failures++; \
    } else { \
        std::cout << "  OK: " << msg << std::endl; \
    } \
} while (0)

// ──────────────────────────────────────────────────────────────────────────
// Test helpers: write synthetic database.h5
// ──────────────────────────────────────────────────────────────────────────

static void write_1d_double_dset(hid_t loc, const char* name,
                                  const double* data, hsize_t n) {
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t dset = H5Dcreate(loc, name, H5T_NATIVE_DOUBLE,
                            space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Sclose(space);
    H5Dclose(dset);
}

static void write_2d_double_dset(hid_t loc, const char* name,
                                  const double* data, hsize_t r, hsize_t c) {
    hsize_t dims[2] = {r, c};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset = H5Dcreate(loc, name, H5T_NATIVE_DOUBLE,
                            space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Sclose(space);
    H5Dclose(dset);
}

static void write_int_scalar(hid_t loc, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dset = H5Dcreate(loc, name, H5T_NATIVE_INT,
                            space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Sclose(space);
    H5Dclose(dset);
}

static void write_string_1d(hid_t loc, const char* name,
                             const char* const* strings, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(dtype, H5T_VARIABLE);
    hid_t dset = H5Dcreate(loc, name, dtype,
                            space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, strings);
    H5Tclose(dtype);
    H5Sclose(space);
    H5Dclose(dset);
}

static void create_group(hid_t loc, const char* name) {
    hid_t grp = H5Gcreate(loc, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(grp);
}

// ──────────────────────────────────────────────────────────────────────────
// Main test
// ──────────────────────────────────────────────────────────────────────────

int main() {
    Kokkos::initialize();
    {
        const char* test_file = "/tmp/test_datacache_db.h5";
        const int maxlag = 5;

        // ── Build synthetic database.h5 ────────────────────────────────────
        {
            hid_t fid = H5Fcreate(test_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // /index — 2 phases, 1 station
            create_group(fid, "index");
            hid_t idx_grp = H5Gopen(fid, "index", H5P_DEFAULT);
            {
                const char* phase_ids[] = {"NET.ST1.Z.P", "NET.ST1.Z.S"};
                write_string_1d(idx_grp, "phase_ids", phase_ids, 2);

                const char* phase_type[] = {"P", "S"};
                write_string_1d(idx_grp, "phase_type", phase_type, 2);

                int station_idx[] = {0, 0};
                write_1d_double_dset(idx_grp, "station_idx",
                                     reinterpret_cast<const double*>(station_idx), 2);

                double dist[] = {100.0, 100.0};
                write_1d_double_dset(idx_grp, "distance", dist, 2);

                double az[] = {45.0, 45.0};
                write_1d_double_dset(idx_grp, "azimuth", az, 2);

                // greens_depth_idx: [2 phases × 1 depth] = [0, 0]
                // Written as 1D of 2 elements (flat)
                double gidx[] = {0.0, 0.0};
                write_1d_double_dset(idx_grp, "greens_depth_idx", gidx, 2);
            }
            H5Gclose(idx_grp);

            // /data/0/XCorr/NET.ST1.Z.P/
            create_group(fid, "data");
            hid_t data_grp = H5Gopen(fid, "data", H5P_DEFAULT);

            create_group(data_grp, "0");
            hid_t freq_grp = H5Gopen(data_grp, "0", H5P_DEFAULT);

            create_group(freq_grp, "XCorr");
            hid_t xc_grp = H5Gopen(freq_grp, "XCorr", H5P_DEFAULT);

            create_group(xc_grp, "NET.ST1.Z.P");
            hid_t phase_grp = H5Gopen(xc_grp, "NET.ST1.Z.P", H5P_DEFAULT);

            // Known test data: obs = [1, 2, 3], gf = [4,0,0,0,0,0, 5,0,0,0,0,0, 6,0,0,0,0,0]
            // N_samples = 3, only Mxx non-zero
            double obs[] = {1.0, 2.0, 3.0};
            double gf[] = {4.0,0,0,0,0,0,  5.0,0,0,0,0,0,  6.0,0,0,0,0,0};
            write_1d_double_dset(phase_grp, "obs", obs, 3);
            write_1d_double_dset(phase_grp, "gf", gf, 18);
            H5Gclose(phase_grp);

            // Second phase: NET.ST1.Z.S — same values
            create_group(xc_grp, "NET.ST1.Z.S");
            phase_grp = H5Gopen(xc_grp, "NET.ST1.Z.S", H5P_DEFAULT);
            write_1d_double_dset(phase_grp, "obs", obs, 3);
            write_1d_double_dset(phase_grp, "gf", gf, 18);
            H5Gclose(phase_grp);
            H5Gclose(xc_grp);

            // Polarity
            create_group(freq_grp, "Polarity");
            hid_t pol_grp = H5Gopen(freq_grp, "Polarity", H5P_DEFAULT);
            create_group(pol_grp, "NET.ST1.Z.P");
            phase_grp = H5Gopen(pol_grp, "NET.ST1.Z.P", H5P_DEFAULT);

            // gf_pol: [2 × 6] — 2 time steps, only Mxx non-zero
            double gf_pol[] = {1.0,0,0,0,0,0,  2.0,0,0,0,0,0};
            write_2d_double_dset(phase_grp, "gf_pol", gf_pol, 2, 6);
            write_int_scalar(phase_grp, "obs_pol", 1);
            H5Gclose(phase_grp);

            create_group(pol_grp, "NET.ST1.Z.S");
            phase_grp = H5Gopen(pol_grp, "NET.ST1.Z.S", H5P_DEFAULT);
            write_2d_double_dset(phase_grp, "gf_pol", gf_pol, 2, 6);
            write_int_scalar(phase_grp, "obs_pol", -1);
            H5Gclose(phase_grp);
            H5Gclose(pol_grp);

            // PSR
            create_group(freq_grp, "PSR");
            hid_t psr_grp = H5Gopen(freq_grp, "PSR", H5P_DEFAULT);
            create_group(psr_grp, "NET.ST1.Z.P");
            phase_grp = H5Gopen(psr_grp, "NET.ST1.Z.P", H5P_DEFAULT);

            // amp_P and amp_S: identity-ish 6×6
            double amp[36] = {};
            double ampS[36] = {};
            for (int i = 0; i < 6; ++i) { amp[i*6+i] = 1.0; ampS[i*6+i] = 2.0; }
            write_2d_double_dset(phase_grp, "amp_P", amp, 6, 6);
            write_2d_double_dset(phase_grp, "amp_S", ampS, 6, 6);
            write_1d_double_dset(phase_grp, "obs_psr", const_cast<double*>(const_cast<const double*>(&(double){0.5})), 1);
            H5Gclose(phase_grp);

            create_group(psr_grp, "NET.ST1.Z.S");
            phase_grp = H5Gopen(psr_grp, "NET.ST1.Z.S", H5P_DEFAULT);
            write_2d_double_dset(phase_grp, "amp_P", amp, 6, 6);
            write_2d_double_dset(phase_grp, "amp_S", ampS, 6, 6);
            write_1d_double_dset(phase_grp, "obs_psr", const_cast<double*>(const_cast<const double*>(&(double){0.3})), 1);
            H5Gclose(phase_grp);
            H5Gclose(psr_grp);

            H5Gclose(freq_grp);
            H5Gclose(data_grp);
            H5Fclose(fid);
        }

        std::cout << "Setup: synthetic database written." << std::endl;

        // ── Test DataCache ─────────────────────────────────────────────────
        {
            DataCache cache(maxlag);

            // Trials referencing the one combo
            std::vector<Trial> trials = {
                {45.0, 30.0, 0.0, 10.0, 0, 0},
                {47.0, 30.0, 0.0, 10.0, 0, 0},
            };

            cache.load_from_database(test_file, trials);

            // Verify cache size
            CHECK(cache.size() == 1, "cache size = 1 (one unique combo)");

            // Retrieve the entry
            const CacheEntry* entry = cache.get_or_compute(0, 0);
            CHECK(entry != nullptr, "get_or_compute(0,0) returns entry");
            CHECK(entry->valid(), "entry is valid");
            CHECK(entry->freq_idx == 0, "freq_idx == 0");
            CHECK(entry->depth_idx == 0, "depth_idx == 0");
            CHECK(entry->n_phases == 2, "n_phases == 2");
            CHECK(entry->maxlag == maxlag, "maxlag preserved");

            // ── Verify XCorr cache ─────────────────────────────────────────
            // Host mirrors for inspection
            auto cc_h = Kokkos::create_mirror_view(entry->xcorr.cc);
            Kokkos::deep_copy(cc_h, entry->xcorr.cc);
            auto synamp_h = Kokkos::create_mirror_view(entry->xcorr.synamp);
            Kokkos::deep_copy(synamp_h, entry->xcorr.synamp);
            auto obs_norm2_h = Kokkos::create_mirror_view(entry->xcorr.obs_norm2);
            Kokkos::deep_copy(obs_norm2_h, entry->xcorr.obs_norm2);

            // obs_norm2 = sum(obs^2) = 1^2 + 2^2 + 3^2 = 14
            CHECK_CLOSE(obs_norm2_h(0), 14.0, 1e-10, "obs_norm2 phase 0 = 14");
            CHECK_CLOSE(obs_norm2_h(1), 14.0, 1e-10, "obs_norm2 phase 1 = 14");

            // synamp[0][0] = sum(gf_col0^2) = 4^2 + 5^2 + 6^2 = 77
            CHECK_CLOSE(synamp_h(0, 0), 77.0, 1e-10, "synamp[0][0] for phase 0 = 77");

            // Cross-correlation at lag=0, comp=0: sum(obs[t] * gf[t,0]) = 1*4 + 2*5 + 3*6 = 32
            int cc_rows = 2 * maxlag + 1;
            CHECK_CLOSE(cc_h(0 * cc_rows + maxlag, 0), 32.0, 1e-10,
                        "CC[lag=0, comp=0] phase 0 = 32");

            // ── Verify Polarity cache ───────────────────────────────────────
            auto pol_vec_h = Kokkos::create_mirror_view(entry->polarity.pol_vec);
            Kokkos::deep_copy(pol_vec_h, entry->polarity.pol_vec);
            auto obs_pol_h = Kokkos::create_mirror_view(entry->polarity.obs_pol);
            Kokkos::deep_copy(obs_pol_h, entry->polarity.obs_pol);

            // gf_pol sum for comp 0: 1 + 2 = 3, comp 1-5: 0
            CHECK_CLOSE(pol_vec_h(0 * 6 + 0), 3.0, 1e-10, "pol_vec[0] phase 0 = 3");
            CHECK_CLOSE(pol_vec_h(0 * 6 + 1), 0.0, 1e-10, "pol_vec[1] phase 0 = 0");
            CHECK_CLOSE(obs_pol_h(0), 1.0, 1e-10, "obs_pol phase 0 = 1");
            CHECK_CLOSE(obs_pol_h(1), -1.0, 1e-10, "obs_pol phase 1 = -1");

            // ── Verify PSR cache ────────────────────────────────────────────
            auto ampP_h = Kokkos::create_mirror_view(entry->psr.amp_P);
            Kokkos::deep_copy(ampP_h, entry->psr.amp_P);
            auto obs_psr_h = Kokkos::create_mirror_view(entry->psr.obs_psr);
            Kokkos::deep_copy(obs_psr_h, entry->psr.obs_psr);

            CHECK_CLOSE(ampP_h(0, 0), 1.0, 1e-10, "amp_P[0][0] phase 0 = 1");
            CHECK_CLOSE(obs_psr_h(0), 0.5, 1e-10, "obs_psr phase 0 = 0.5");
            CHECK_CLOSE(obs_psr_h(1), 0.3, 1e-10, "obs_psr phase 1 = 0.3");

            // ── Test non-existent combo throws ──────────────────────────────
            bool threw = false;
            try {
                cache.get_or_compute(99, 99);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            CHECK(threw, "get_or_compute(unknown) throws");

            // ── Test release_all ────────────────────────────────────────────
            cache.release_all();
            CHECK(cache.size() == 0, "cache empty after release_all");
        }

        std::cout << std::endl;
    }
    Kokkos::finalize();

    if (failures == 0) {
        std::cout << "All DataCache tests passed." << std::endl;
        return 0;
    } else {
        std::cerr << failures << " test(s) FAILED." << std::endl;
        return 1;
    }
}