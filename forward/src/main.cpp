#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <hdf5.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda_runtime.h"
#include "backends/device.h"
#include "data_cache.h"
#include "hdf5_io.h"
#include "intermediates_transaction.h"
#include "kernels/polarity_kernel.h"
#include "kernels/xcorr_kernel.h"
#include "mt_utils.h"
#include "validation.h"

// main — forward stage entry point: forward <database.h5> <status_N.h5>
// runs the misfit kernels and writes raw intermediates to
// status_N.h5:/intermediates/{Module}/ (final extract/compose is Julia assess.jl).

// Per-module config metadata read from database.h5:/config/{Module}/
struct ModuleConfig {
    std::string name;
    std::string op;      // "Xcorr" / "Polarity"
    std::string phase;   // "P" / "S"
    std::string channel; // "" or "H"/"V"
    bool is_composed;
};

struct XCorrConfigContract {
    std::string module;
    double max_lag_periods;
    std::vector<int> band_low;
    std::vector<int> band_high;
    std::vector<double> trim;
};

enum class BackendRequest { Auto, Cpu, Cuda };

struct CommandLineOptions {
    BackendRequest backend;
    std::optional<size_t> cuda_batch_trials;
    std::string database_path;
    std::string status_path;
};

struct BackendSelection {
    Backend backend;
    std::string device_name;
    double cuda_context_ms;
};

constexpr const char *USAGE = "Usage: forward [--backend auto|cpu|cuda] [--cuda-batch-trials N] "
                              "<database.h5> <status_N.h5>";

BackendRequest parse_backend(const std::string &value) {
    if (value == "auto")
        return BackendRequest::Auto;
    if (value == "cpu")
        return BackendRequest::Cpu;
    if (value == "cuda")
        return BackendRequest::Cuda;
    throw std::runtime_error("invalid backend: " + value);
}

const char *backend_name(BackendRequest backend) {
    if (backend == BackendRequest::Cpu)
        return "cpu";
    if (backend == BackendRequest::Cuda)
        return "cuda";
    return "auto";
}

size_t parse_positive_size(const std::string &value, const char *option) {
    if (value.empty() || value.front() == '-')
        throw std::runtime_error(std::string(option) + " must be a positive integer");
    size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || parsed == 0)
        throw std::runtime_error(std::string(option) + " must be a positive integer");
    return parsed;
}

CommandLineOptions parse_command_line(int argc, char *argv[]) {
    std::string backend_value = "auto";
    if (const char *environment = std::getenv("FM_FORWARD_BACKEND"))
        backend_value = environment;
    std::optional<std::string> batch_value;
    if (const char *environment = std::getenv("FM_CUDA_BATCH_TRIALS"))
        batch_value = environment;

    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--backend") {
            if (++index >= argc)
                throw std::runtime_error("--backend requires a value");
            backend_value = argv[index];
        } else if (argument == "--cuda-batch-trials") {
            if (++index >= argc)
                throw std::runtime_error("--cuda-batch-trials requires a value");
            batch_value = argv[index];
        } else if (argument.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option: " + argument);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() != 2)
        throw std::runtime_error(USAGE);

    std::optional<size_t> batch_trials;
    if (batch_value)
        batch_trials = parse_positive_size(*batch_value, "--cuda-batch-trials");
    return {parse_backend(backend_value), batch_trials, positional[0], positional[1]};
}

BackendSelection select_backend(BackendRequest request) {
    if (request == BackendRequest::Cpu)
        return {Backend::OpenMP, "", 0.0};

    const fm::CudaProbeResult probe = fm::probe_cuda_device();
    if (probe.status == fm::CudaProbeStatus::Available)
        return {Backend::CUDA, probe.device_name, probe.initialization_ms};
    if (request == BackendRequest::Auto && (probe.status == fm::CudaProbeStatus::NoDevice ||
                                            probe.status == fm::CudaProbeStatus::NotCompiled))
        return {Backend::OpenMP, "", probe.initialization_ms};
    if (probe.status == fm::CudaProbeStatus::NotCompiled)
        throw std::runtime_error("CUDA backend not compiled");
    if (probe.status == fm::CudaProbeStatus::NoDevice)
        throw std::runtime_error("no CUDA device available");
    throw std::runtime_error("CUDA initialization failed: " + probe.error);
}

std::string intermediate_key(const ModuleConfig &module) {
    std::string key = module.op + module.phase;
    if (!module.channel.empty())
        key += "_" + module.channel;
    return key;
}

void validate_intermediates_group(Hdf5Handle &file, const std::string &root,
                                  const std::vector<ModuleConfig> &modules, int n_p, int n_s,
                                  int n_stations, int n_trials) {
    if (!file.group_exists(root.c_str()))
        throw std::runtime_error(root + ": group does not exist");

    std::set<std::string> validated_keys;
    for (const auto &module : modules) {
        if (module.is_composed)
            continue;
        const std::string key = intermediate_key(module);
        if (!validated_keys.insert(key).second)
            continue;
        const std::string group = root + "/" + key;
        if (!file.group_exists(group.c_str()))
            throw std::runtime_error(group + ": group does not exist");
        if (module.op == "Xcorr" && module.phase == "P") {
            file.validate_dataset_2d((group + "/cc_max").c_str(), H5T_NATIVE_DOUBLE, n_p, n_trials);
            file.validate_dataset_2d((group + "/best_lag").c_str(), H5T_NATIVE_INT32, n_p,
                                     n_trials);
        } else if (module.op == "Xcorr" && module.phase == "S") {
            file.validate_dataset_2d((group + "/cc_max").c_str(), H5T_NATIVE_DOUBLE, n_s, n_trials);
            file.validate_dataset_2d((group + "/best_lag").c_str(), H5T_NATIVE_INT32, n_s,
                                     n_trials);
        } else if (module.op == "Polarity") {
            file.validate_dataset_2d((group + "/syn_sign").c_str(), H5T_NATIVE_INT8, n_stations,
                                     n_trials);
            file.validate_dataset_2d((group + "/dot_value").c_str(), H5T_NATIVE_DOUBLE, n_stations,
                                     n_trials);
        } else {
            throw std::runtime_error("unsupported intermediate module: " + module.name);
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        const auto forward_started = std::chrono::steady_clock::now();
        const CommandLineOptions options = parse_command_line(argc, argv);
        const BackendSelection selection = select_backend(options.backend);
        if (options.cuda_batch_trials && selection.backend != Backend::CUDA)
            throw std::runtime_error("--cuda-batch-trials requires CUDA backend");

        std::cout << "fm_forward: requested backend=" << backend_name(options.backend)
                  << ", selected backend=" << (selection.backend == Backend::CUDA ? "cuda" : "cpu");
        if (selection.backend == Backend::CUDA)
            std::cout << ", device=0 (" << selection.device_name << ")";
        std::cout << std::endl;

        const std::string &database_path = options.database_path;
        const std::string &status_path = options.status_path;

        // 1. Read trials from status_N.h5
        Hdf5Handle status_file;
        status_file.open(status_path.c_str(), H5F_ACC_RDWR);

        // Physical axis values from /paraspace (sole authority); trials carry indices
        Hdf5Handle db_reader;
        db_reader.open(database_path.c_str(), H5F_ACC_RDONLY);

        int N_trials = status_file.read_int_scalar("/trials/N_trials");

        auto s_idx_host = status_file.read_int_1d("/trials/strike_idx");
        auto d_idx_host = status_file.read_int_1d("/trials/dip_idx");
        auto r_idx_host = status_file.read_int_1d("/trials/rake_idx");
        auto dep_idx_host = status_file.read_int_1d("/trials/depth_idx");
        auto f_idx_host = status_file.read_int_1d("/trials/freq_idx");
        auto duration_idx_host = status_file.read_int_1d("/trials/duration_idx");

        auto strike_vals = db_reader.read_double_1d("/paraspace/strike");
        auto dip_vals = db_reader.read_double_1d("/paraspace/dip");
        auto rake_vals = db_reader.read_double_1d("/paraspace/rake");
        auto depth_vals = db_reader.read_double_1d("/paraspace/depth");
        auto frequency_vals = db_reader.read_double_1d("/paraspace/frequency");
        auto duration_vals = db_reader.read_double_1d("/paraspace/duration");

        fm::validate_trial_indices(
            N_trials,
            fm::TrialIndexView(s_idx_host, d_idx_host, r_idx_host, dep_idx_host, f_idx_host,
                               duration_idx_host),
            fm::ParaspaceSizes {strike_vals.size(), dip_vals.size(), rake_vals.size(),
                                depth_vals.size(), frequency_vals.size(), duration_vals.size()});
        fm::validate_finite(strike_vals.data(), strike_vals.size(), "/paraspace/strike");
        fm::validate_finite(dip_vals.data(), dip_vals.size(), "/paraspace/dip");
        fm::validate_finite(rake_vals.data(), rake_vals.size(), "/paraspace/rake");
        fm::validate_finite(depth_vals.data(), depth_vals.size(), "/paraspace/depth");
        fm::validate_finite(frequency_vals.data(), frequency_vals.size(), "/paraspace/frequency");
        fm::validate_finite(duration_vals.data(), duration_vals.size(), "/paraspace/duration");

        const size_t trial_count = static_cast<size_t>(N_trials);
        std::vector<Trial> trials(trial_count);
        for (int i = 0; i < N_trials; ++i) {
            int32_t si = static_cast<int32_t>(s_idx_host[i]);
            int32_t di = static_cast<int32_t>(d_idx_host[i]);
            int32_t ri = static_cast<int32_t>(r_idx_host[i]);
            int32_t depi = static_cast<int32_t>(dep_idx_host[i]);
            int32_t fi = static_cast<int32_t>(f_idx_host[i]);
            int32_t dui = static_cast<int32_t>(duration_idx_host[i]);
            trials[i] = Trial {si,
                               di,
                               ri,
                               depi,
                               fi,
                               dui,
                               strike_vals[static_cast<size_t>(si - 1)],
                               dip_vals[static_cast<size_t>(di - 1)],
                               rake_vals[static_cast<size_t>(ri - 1)]};
        }

        // 2. SDR → MT conversion (host-side, degrees to radians)
        // XCorr uses [6 × N_trials] (LayoutLeft), Polarity uses [N_trials × 6]
        const size_t mt_count = fm::checked_mul(trial_count, 6, "moment tensor");
        std::vector<double> mt_xcorr_host(mt_count);
        std::vector<double> mt_pol_host(mt_count);

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

        // 3. Read module config from database.h5:/config (db_reader open above)
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

        bool has_xcorr_config = false;
        XCorrConfigContract xcorr_config;
        for (const auto &mc : modules) {
            if (mc.is_composed || mc.op != "Xcorr")
                continue;
            if (mc.phase != "P" && mc.phase != "S")
                throw std::runtime_error("XCorr module has unsupported phase: " + mc.name);

            const std::string base = "/config/" + mc.name + "/";
            XCorrConfigContract current {
                mc.name, db_reader.read_double_scalar((base + "max_lag_periods").c_str()),
                db_reader.read_int_1d((base + "band_low").c_str()),
                db_reader.read_int_1d((base + "band_high").c_str()),
                db_reader.read_double_1d((base + "trim").c_str())};
            fm::validate_finite(&current.max_lag_periods, 1, base + "max_lag_periods");
            fm::validate_finite(current.trim.data(), current.trim.size(), base + "trim");
            if (current.max_lag_periods <= 0.0 || current.band_low.empty() ||
                current.band_high.empty() || current.band_low.size() != current.band_high.size() ||
                current.trim.size() != 2)
                throw std::runtime_error("invalid XCorr configuration for module " + mc.name);

            if (!has_xcorr_config) {
                xcorr_config = std::move(current);
                has_xcorr_config = true;
            } else if (current.max_lag_periods != xcorr_config.max_lag_periods ||
                       current.band_low != xcorr_config.band_low ||
                       current.band_high != xcorr_config.band_high ||
                       current.trim != xcorr_config.trim) {
                throw std::runtime_error("XCorr P/S configuration mismatch between " +
                                         xcorr_config.module + " and " + current.module);
            }
        }
        if (!has_xcorr_config)
            throw std::runtime_error("no active XCorr module configured");

        if (selection.backend == Backend::CUDA) {
            for (const auto &module : modules) {
                if (!module.is_composed && module.op != "Xcorr")
                    throw std::runtime_error("CUDA backend supports only active XCorr operators");
            }
        }

        // Read P/S station indices (data partitioned by XcorrP/XcorrS groups)
        int n_p = 0, n_s = 0;
        std::vector<int> st_idx_vec;
        if (db_reader.group_exists("/XcorrP/station_idx")) {
            auto p_si = db_reader.read_int_1d("/XcorrP/station_idx");
            for (auto &v : p_si)
                --v; // 1-based in HDF5 -> 0-based vector index
            n_p = static_cast<int>(p_si.size());
            st_idx_vec.insert(st_idx_vec.end(), p_si.begin(), p_si.end());
        }
        if (db_reader.group_exists("/XcorrS/station_idx")) {
            auto s_si = db_reader.read_int_1d("/XcorrS/station_idx");
            for (auto &v : s_si)
                --v; // 1-based in HDF5 -> 0-based vector index
            n_s = static_cast<int>(s_si.size());
            st_idx_vec.insert(st_idx_vec.end(), s_si.begin(), s_si.end());
        }
        const size_t phase_count =
            fm::checked_add(static_cast<size_t>(n_p), static_cast<size_t>(n_s), "phase count");
        if (phase_count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("phase count exceeds INT_MAX");
        int N_phases = static_cast<int>(phase_count);

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

        auto validate_intermediates = [&](const std::string &root) {
            validate_intermediates_group(status_file, root, modules, n_p, n_s, N_stations,
                                         N_trials);
        };
        fm::recover_intermediates(status_file, validate_intermediates);

        // maxlag = round(max_lag_periods / band_high / dt); window-clamped per combo in DataCache.
        for (int index : xcorr_config.band_low)
            if (index < 1 || static_cast<size_t>(index) > frequency_vals.size())
                throw std::runtime_error("XCorr band_low index out of range");
        for (int index : xcorr_config.band_high)
            if (index < 1 || static_cast<size_t>(index) > frequency_vals.size())
                throw std::runtime_error("XCorr band_high index out of range");
        const double band_high_freq =
            frequency_vals[static_cast<size_t>(xcorr_config.band_high[0] - 1)];
        auto dt_arr = db_reader.read_double_1d("/station/dt");
        fm::validate_finite(dt_arr.data(), dt_arr.size(), "/station/dt");
        if (dt_arr.empty() || band_high_freq <= 0.0)
            throw std::runtime_error("invalid XCorr sampling or frequency configuration");
        const double dt = dt_arr[0];
        if (dt <= 0.0)
            throw std::runtime_error("station dt must be positive");
        for (int station_index : st_idx_vec) {
            if (station_index < 0 || static_cast<size_t>(station_index) >= dt_arr.size())
                throw std::runtime_error("XCorr station index out of range");
            if (dt_arr[static_cast<size_t>(station_index)] != dt)
                throw std::runtime_error("XCorr P/S sampling interval mismatch");
        }
        const int maxlag = std::max(
            1, static_cast<int>(std::llround(xcorr_config.max_lag_periods / band_high_freq / dt)));

        db_reader.close();

        // 4. Initialize DataCache, load preprocessed data
        DataCache cache(maxlag);
        cache.load_from_database(database_path, trials);

        // Collect unique (freq_idx, depth_idx, duration_idx) combos from trials
        std::set<CacheKey> combo_set;
        for (const auto &t : trials)
            combo_set.insert({t.freq_idx, t.depth_idx, t.duration_idx});
        std::vector<CacheKey> combos(combo_set.begin(), combo_set.end());

        size_t max_cc_rows = 0;
        for (const auto &[f_idx, d_idx, duration_idx] : combos) {
            const CacheEntry *entry = cache.get_or_compute(f_idx, d_idx, duration_idx);
            if (!entry->valid() || entry->xcorr.cc == nullptr || entry->xcorr.synamp == nullptr ||
                entry->xcorr.obs_norm2 == nullptr)
                throw std::runtime_error("invalid XCorr cache entry for combo");
            if (entry->xcorr.n_phases != N_phases || entry->xcorr.cc_rows <= 0 ||
                entry->xcorr.maxlag < 0 || entry->xcorr.cc_rows != 2 * entry->xcorr.maxlag + 1)
                throw std::runtime_error("invalid XCorr cache shape for combo");

            const size_t cc_count = fm::checked_mul(
                fm::checked_mul(phase_count, static_cast<size_t>(entry->xcorr.cc_rows),
                                "XCorr cc rows"),
                6, "XCorr cc components");
            const size_t synamp_count =
                fm::checked_mul(fm::checked_mul(phase_count, 36, "XCorr synamp phases"),
                                static_cast<size_t>(entry->xcorr.cc_rows), "XCorr synamp lags");
            fm::validate_finite(entry->xcorr.cc, cc_count, "XCorr cc");
            fm::validate_finite(entry->xcorr.synamp, synamp_count, "XCorr synamp");
            fm::validate_finite(entry->xcorr.obs_norm2, phase_count, "XCorr obs_norm2");
            max_cc_rows = std::max(max_cc_rows, static_cast<size_t>(entry->xcorr.cc_rows));
        }

        std::unique_ptr<fm::CudaXcorrExecutor> cuda_executor;
        if (selection.backend == Backend::CUDA) {
            cuda_executor = std::make_unique<fm::CudaXcorrExecutor>(
                phase_count, max_cc_rows, trial_count, options.cuda_batch_trials);
            std::cout << "fm_forward: CUDA batch capacity=" << cuda_executor->batch_capacity()
                      << std::endl;
        }

        // 5. Allocate intermediate output arrays (accumulated across combos)
        bool has_xcorr_p = n_p > 0;
        bool has_xcorr_s = n_s > 0;
        bool has_polarity = N_stations > 0;

        const size_t p_output_count =
            has_xcorr_p ? fm::checked_mul(static_cast<size_t>(n_p), trial_count, "P output") : 0;
        const size_t s_output_count =
            has_xcorr_s ? fm::checked_mul(static_cast<size_t>(n_s), trial_count, "S output") : 0;
        const size_t station_output_count =
            has_polarity
                ? fm::checked_mul(static_cast<size_t>(N_stations), trial_count, "station output")
                : 0;
        std::vector<double> cc_max_p(p_output_count, 0.0);
        std::vector<int32_t> best_lag_p(p_output_count, 0);
        std::vector<double> cc_max_s(s_output_count, 0.0);
        std::vector<int32_t> best_lag_s(s_output_count, 0);
        std::vector<int8_t> syn_sign(station_output_count, 0);
        std::vector<double> dot_value(station_output_count,
                                      std::numeric_limits<double>::quiet_NaN());
        std::vector<uint8_t> completed(trial_count, 0);

        // 6. Launch kernels per combo, accumulate into intermediate arrays
        const auto evaluation_started = std::chrono::steady_clock::now();
        for (const auto &combo : combos) {
            const auto [f_idx, d_idx, duration_idx] = combo;

            std::vector<int> trial_indices;
            for (int t = 0; t < N_trials; ++t)
                if (trials[t].freq_idx == f_idx && trials[t].depth_idx == d_idx &&
                    trials[t].duration_idx == duration_idx)
                    trial_indices.push_back(t);
            if (trial_indices.empty())
                throw std::runtime_error("empty trial combo");

            int n_sub = static_cast<int>(trial_indices.size());
            const CacheEntry *entry = cache.get_or_compute(f_idx, d_idx, duration_idx);
            if (!entry || !entry->valid())
                throw std::runtime_error("invalid cache entry during combo evaluation");

            // ── Build MT sub-views for this combo's trials ──
            const size_t mt_sub_count =
                fm::checked_mul(static_cast<size_t>(n_sub), 6, "combo moment tensor");
            std::vector<double> mt_xcorr_sub(mt_sub_count);
            std::vector<double> mt_pol_sub(mt_sub_count);
            for (int si = 0; si < n_sub; ++si) {
                int ti = trial_indices[si];
                for (int c = 0; c < 6; ++c) {
                    mt_xcorr_sub[c + si * 6] = mt_xcorr_host[c + ti * 6];
                    mt_pol_sub[si + c * n_sub] = mt_pol_host[ti + c * N_trials];
                }
            }

            // ── XCorr: cc_max + best_lag per (phase, trial) ──
            if (entry->xcorr.cc != nullptr && N_phases > 0) {
                // Per-lag synamp lives in the cache entry already (see data_cache.h).
                const int cc_pp = entry->xcorr.cc_rows;   // 2*maxlag+1
                const int maxlag_e = entry->xcorr.maxlag; // window-clamped half-width

                const size_t combo_output_count =
                    fm::checked_mul(phase_count, static_cast<size_t>(n_sub), "combo XCorr output");
                if (combo_output_count > static_cast<size_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("combo XCorr work-item count exceeds INT_MAX");
                std::vector<double> cc_max_sub(combo_output_count);
                std::vector<int32_t> best_lag_sub(combo_output_count);

                if (selection.backend == Backend::CUDA) {
                    const std::string combo_context = "combo freq=" + std::to_string(f_idx) +
                                                      " depth=" + std::to_string(d_idx) +
                                                      " duration=" + std::to_string(duration_idx);
                    cuda_executor->evaluate(mt_xcorr_sub.data(), entry->xcorr.cc,
                                            entry->xcorr.synamp, entry->xcorr.obs_norm2,
                                            cc_max_sub.data(), best_lag_sub.data(),
                                            static_cast<size_t>(n_sub), static_cast<size_t>(cc_pp),
                                            maxlag_e, combo_context);
                } else {
                    fm::launch_xcorr_openmp(mt_xcorr_sub.data(), entry->xcorr.cc,
                                            entry->xcorr.synamp, entry->xcorr.obs_norm2,
                                            cc_max_sub.data(), best_lag_sub.data(), N_phases, n_sub,
                                            cc_pp, maxlag_e);
                }

                // Write back: split P (rows 0..n_p) and S (rows n_p..N_phases)
                for (int ph = 0; ph < N_phases; ++ph)
                    for (int si = 0; si < n_sub; ++si) {
                        double v = cc_max_sub[ph + si * N_phases];
                        int32_t lag = best_lag_sub[ph + si * N_phases];
                        if (ph < n_p) {
                            const size_t output_index =
                                static_cast<size_t>(ph) * trial_count + trial_indices[si];
                            cc_max_p[output_index] = v;
                            best_lag_p[output_index] = lag;
                        } else {
                            int sp = ph - n_p;
                            const size_t output_index =
                                static_cast<size_t>(sp) * trial_count + trial_indices[si];
                            cc_max_s[output_index] = v;
                            best_lag_s[output_index] = lag;
                        }
                    }
                for (int trial_index : trial_indices)
                    fm::mark_trial_completed(completed, static_cast<size_t>(trial_index));
            } else {
                throw std::runtime_error("XCorr cache missing during combo evaluation");
            }

            // ── Polarity: syn_sign + dot_value per (station, trial) ──
            if (entry->polarity.pol_vec != nullptr && N_stations > 0) {
                const size_t polarity_vector_count =
                    fm::checked_mul(static_cast<size_t>(N_stations), 6, "polarity vector");
                const size_t polarity_output_count =
                    fm::checked_mul(static_cast<size_t>(N_stations), static_cast<size_t>(n_sub),
                                    "combo polarity output");
                if (polarity_output_count > static_cast<size_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("combo polarity work-item count exceeds INT_MAX");
                std::vector<double> pol_vec_s(polarity_vector_count);
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

                std::vector<int8_t> syn_sign_sub(polarity_output_count);
                std::vector<double> dot_sub(polarity_output_count);

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
        const double evaluation_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - evaluation_started)
                                         .count();

        fm::validate_trials_completed(completed);

        // 7. Write and atomically promote status_N.h5:/intermediates.
        auto write_intermediates = [&](const std::string &root) {
            status_file.create_group(root.c_str());
            auto write_xcorr_inter = [&](const std::string &key, const std::vector<double> &cc,
                                         const std::vector<int32_t> &lag, int n_ph) {
                if (n_ph == 0)
                    return;
                const std::string group = root + "/" + key;
                status_file.create_group(group.c_str());
                status_file.write_double_2d((group + "/cc_max").c_str(), cc.data(), n_ph, N_trials);
                status_file.write_int32_2d((group + "/best_lag").c_str(), lag.data(), n_ph,
                                           N_trials);
            };

            // Map module -> canonical intermediate key (operator + phase [+ channel]).
            std::set<std::string> written_keys;
            for (const auto &module : modules) {
                if (module.is_composed)
                    continue;
                const std::string key = intermediate_key(module);
                if (!written_keys.insert(key).second)
                    continue;
                if (module.op == "Xcorr" && module.phase == "P")
                    write_xcorr_inter(key, cc_max_p, best_lag_p, n_p);
                else if (module.op == "Xcorr" && module.phase == "S")
                    write_xcorr_inter(key, cc_max_s, best_lag_s, n_s);
                else if (module.op == "Polarity") {
                    const std::string group = root + "/" + key;
                    status_file.create_group(group.c_str());
                    status_file.write_int8_2d((group + "/syn_sign").c_str(), syn_sign.data(),
                                              N_stations, N_trials);
                    status_file.write_double_2d((group + "/dot_value").c_str(), dot_value.data(),
                                                N_stations, N_trials);
                }
            }
        };
        fm::commit_intermediates(status_file, write_intermediates, validate_intermediates);

        status_file.close();
        cache.release_all();

        if (cuda_executor) {
            const fm::CudaTimings &timings = cuda_executor->timings();
            std::cout << "fm_forward: CUDA timing ms context=" << selection.cuda_context_ms
                      << " allocation=" << timings.initialization_ms << " h2d=" << timings.h2d_ms
                      << " kernel=" << timings.kernel_ms << " d2h=" << timings.d2h_ms << std::endl;
        }
        const double total_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - forward_started)
                                    .count();
        std::cout << "fm_forward: timing ms evaluation=" << evaluation_ms << " total=" << total_ms
                  << std::endl;

        std::cout << "fm_forward: " << N_trials << " trials × " << combos.size() << " combos -> "
                  << N_phases << " phases, " << N_stations << " stations"
                  << " (intermediates written)" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "fm_forward error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
