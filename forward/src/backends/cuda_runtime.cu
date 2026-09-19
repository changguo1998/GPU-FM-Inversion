#include "backends/cuda_runtime.h"

#include "backends/cuda_batch.h"
#include "kernels/waveform_kernel.h"
#include "kernels/xcorr_kernel.h"
#include "validation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace fm {
namespace {

void check_cuda(cudaError_t error, const std::string &context) {
    if (error != cudaSuccess)
        throw std::runtime_error(context + ": " + cudaGetErrorString(error));
}

CudaProbeResult cuda_error(cudaError_t error, double initialization_ms) {
    return {CudaProbeStatus::Error, "", cudaGetErrorString(error), initialization_ms};
}

template <typename T> class CudaBuffer {
  public:
    CudaBuffer() = default;
    ~CudaBuffer() {
        if (data_)
            cudaFree(data_);
    }
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    void allocate(size_t count, const std::string &context) {
        if (data_)
            throw std::runtime_error(context + ": CUDA buffer already allocated");
        const size_t bytes = checked_mul(count, sizeof(T), context);
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), bytes), context + " cudaMalloc");
    }

    T *data() {
        return data_;
    }
    const T *data() const {
        return data_;
    }

  private:
    T *data_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }
    ~CudaEvent() {
        cudaEventDestroy(event_);
    }
    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    cudaEvent_t get() const {
        return event_;
    }

  private:
    cudaEvent_t event_ {};
};

template <typename Function> double measure_cuda(const std::string &context, Function &&function) {
    CudaEvent start;
    CudaEvent end;
    check_cuda(cudaEventRecord(start.get()), context + " start event");
    function();
    check_cuda(cudaEventRecord(end.get()), context + " end event");
    check_cuda(cudaEventSynchronize(end.get()), context + " synchronize");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
               context + " elapsed time");
    return static_cast<double>(milliseconds);
}

template <typename T>
void copy_to_device(T *destination, const T *source, size_t count, const std::string &context) {
    const size_t bytes = checked_mul(count, sizeof(T), context);
    check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), context);
}

template <typename T>
void copy_to_host(T *destination, const T *source, size_t count, const std::string &context) {
    const size_t bytes = checked_mul(count, sizeof(T), context);
    check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), context);
}

} // namespace

CudaProbeResult probe_cuda_device() {
    const auto started = std::chrono::steady_clock::now();
    auto elapsed_ms = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    };
    if (std::getenv("FM_CUDA_TEST_PROBE_ERROR"))
        return {CudaProbeStatus::Error, "", "injected CUDA initialization failure", elapsed_ms()};

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaErrorNoDevice || (error == cudaSuccess && device_count == 0))
        return {CudaProbeStatus::NoDevice, "", "no CUDA device", elapsed_ms()};
    if (error != cudaSuccess)
        return cuda_error(error, elapsed_ms());

    error = cudaSetDevice(0);
    if (error != cudaSuccess)
        return cuda_error(error, elapsed_ms());

    cudaDeviceProp properties {};
    error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess)
        return cuda_error(error, elapsed_ms());
    return {CudaProbeStatus::Available, properties.name, "", elapsed_ms()};
}

struct CudaXcorrExecutor::Impl {
    Impl(size_t phase_count, size_t maximum_cc_rows, size_t maximum_samples, size_t total_trials,
         std::optional<size_t> batch_limit, bool compute_energy, bool compute_amplitude)
        : n_phases(phase_count), max_cc_rows(maximum_cc_rows), max_samples(maximum_samples),
          need_energy(compute_energy), need_amplitude(compute_amplitude) {
        const auto started = std::chrono::steady_clock::now();
        if (n_phases == 0 || max_cc_rows == 0)
            throw std::runtime_error("CUDA XCorr executor requires phases and CC rows");
        if (n_phases > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            max_cc_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("CUDA XCorr shape exceeds INT_MAX");

        const size_t cc_count =
            checked_mul(checked_mul(n_phases, max_cc_rows, "CUDA XCorr CC rows"), 6,
                        "CUDA XCorr CC components");
        const size_t synamp_count =
            checked_mul(checked_mul(n_phases, 36, "CUDA XCorr synamp phases"), max_cc_rows,
                        "CUDA XCorr synamp lags");
        cc.allocate(cc_count, "CUDA XCorr CC buffer");
        synamp.allocate(synamp_count, "CUDA XCorr synamp buffer");
        obs_norm2.allocate(n_phases, "CUDA XCorr obs norm buffer");
        if (need_amplitude) {
            if (max_samples == 0 ||
                max_samples > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("CUDA waveform shape is invalid");
            gf.allocate(checked_mul(checked_mul(n_phases, max_samples, "CUDA waveform samples"), 6,
                                    "CUDA waveform components"),
                        "CUDA waveform GF buffer");
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        size_t extra_per_trial_bytes = 0;
        if (need_energy)
            extra_per_trial_bytes = checked_add(
                extra_per_trial_bytes, checked_mul(n_phases, sizeof(double), "CUDA energy bytes"),
                "CUDA extra output bytes");
        if (need_amplitude) {
            extra_per_trial_bytes =
                checked_add(extra_per_trial_bytes,
                            checked_mul(n_phases, sizeof(double), "CUDA amplitude bytes"),
                            "CUDA extra output bytes");
            extra_per_trial_bytes = checked_add(
                extra_per_trial_bytes, checked_mul(n_phases, sizeof(int8_t), "CUDA sign bytes"),
                "CUDA extra output bytes");
        }
        const CudaBatchPlan plan = plan_cuda_batch(free_bytes, total_bytes, n_phases, total_trials,
                                                   batch_limit, extra_per_trial_bytes);
        batch_capacity = plan.capacity;

        mt.allocate(checked_mul(batch_capacity, 6, "CUDA batch MT count"), "CUDA batch MT buffer");
        cc_max.allocate(checked_mul(n_phases, batch_capacity, "CUDA batch CC count"),
                        "CUDA batch CC output buffer");
        best_lag.allocate(checked_mul(n_phases, batch_capacity, "CUDA batch lag count"),
                          "CUDA batch lag output buffer");
        if (need_energy)
            energy.allocate(checked_mul(n_phases, batch_capacity, "CUDA batch energy count"),
                            "CUDA batch energy output buffer");
        if (need_amplitude) {
            amp_scale.allocate(checked_mul(n_phases, batch_capacity, "CUDA batch amplitude count"),
                               "CUDA batch amplitude output buffer");
            sign_scale.allocate(checked_mul(n_phases, batch_capacity, "CUDA batch sign count"),
                                "CUDA batch sign output buffer");
        }
        timing.initialization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
    }

    size_t n_phases;
    size_t max_cc_rows;
    size_t max_samples;
    bool need_energy;
    bool need_amplitude;
    size_t batch_capacity = 0;
    CudaTimings timing;
    CudaBuffer<double> cc;
    CudaBuffer<double> synamp;
    CudaBuffer<double> obs_norm2;
    CudaBuffer<double> gf;
    CudaBuffer<double> mt;
    CudaBuffer<double> cc_max;
    CudaBuffer<int32_t> best_lag;
    CudaBuffer<double> energy;
    CudaBuffer<double> amp_scale;
    CudaBuffer<int8_t> sign_scale;
};

CudaXcorrExecutor::CudaXcorrExecutor(size_t n_phases, size_t max_cc_rows, size_t max_samples,
                                     size_t total_trials, std::optional<size_t> batch_limit,
                                     bool need_energy, bool need_amplitude)
    : impl_(new Impl(n_phases, max_cc_rows, max_samples, total_trials, batch_limit, need_energy,
                     need_amplitude)) {
}

CudaXcorrExecutor::~CudaXcorrExecutor() {
    delete impl_;
}

CudaXcorrExecutor::CudaXcorrExecutor(CudaXcorrExecutor &&other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

CudaXcorrExecutor &CudaXcorrExecutor::operator=(CudaXcorrExecutor &&other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

size_t CudaXcorrExecutor::batch_capacity() const {
    if (!impl_)
        throw std::runtime_error("CUDA XCorr executor is not initialized");
    return impl_->batch_capacity;
}

const CudaTimings &CudaXcorrExecutor::timings() const {
    if (!impl_)
        throw std::runtime_error("CUDA XCorr executor is not initialized");
    return impl_->timing;
}

void CudaXcorrExecutor::evaluate(const double *mt_host, const double *cc_host,
                                 const double *synamp_host, const double *obs_norm2_host,
                                 const double *gf_host, double *cc_max_host, int32_t *best_lag_host,
                                 double *energy_host, double *amp_scale_host,
                                 int8_t *sign_scale_host, size_t n_trials, size_t cc_rows,
                                 size_t n_samples, int maxlag, const std::string &context) {
    if (!impl_)
        throw std::runtime_error(context + ": CUDA XCorr executor is not initialized");
    if (!mt_host || !cc_host || !synamp_host || !obs_norm2_host || !cc_max_host || !best_lag_host)
        throw std::runtime_error(context + ": null CUDA XCorr pointer");
    if (impl_->need_energy && !energy_host)
        throw std::runtime_error(context + ": missing CUDA energy output");
    if (impl_->need_amplitude && (!gf_host || !amp_scale_host || !sign_scale_host))
        throw std::runtime_error(context + ": missing CUDA waveform input or output");
    if (n_trials == 0 || cc_rows == 0 || cc_rows > impl_->max_cc_rows)
        throw std::runtime_error(context + ": invalid CUDA XCorr shape");
    if (cc_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(context + ": CUDA XCorr rows exceed INT_MAX");
    if (impl_->need_amplitude && (n_samples == 0 || n_samples > impl_->max_samples))
        throw std::runtime_error(context + ": invalid CUDA waveform shape");

    const size_t cc_count = checked_mul(checked_mul(impl_->n_phases, cc_rows, context + " CC rows"),
                                        6, context + " CC components");
    const size_t synamp_count =
        checked_mul(checked_mul(impl_->n_phases, 36, context + " synamp phases"), cc_rows,
                    context + " synamp lags");
    impl_->timing.h2d_ms += measure_cuda(context + " combo H2D", [&]() {
        copy_to_device(impl_->cc.data(), cc_host, cc_count, context + " CC H2D");
        copy_to_device(impl_->synamp.data(), synamp_host, synamp_count, context + " synamp H2D");
        copy_to_device(impl_->obs_norm2.data(), obs_norm2_host, impl_->n_phases,
                       context + " obs norm H2D");
        if (impl_->need_amplitude) {
            const size_t gf_count =
                checked_mul(checked_mul(impl_->n_phases, n_samples, context + " waveform samples"),
                            6, context + " waveform components");
            copy_to_device(impl_->gf.data(), gf_host, gf_count, context + " waveform H2D");
        }
    });

    std::optional<size_t> injected_failure_batch;
    if (const char *value = std::getenv("FM_CUDA_TEST_FAIL_BATCH")) {
        size_t parsed = 0;
        const std::string text(value);
        const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
        if (result.ec != std::errc() || result.ptr != text.data() + text.size())
            throw std::runtime_error("FM_CUDA_TEST_FAIL_BATCH must be a non-negative integer");
        injected_failure_batch = parsed;
    }

    size_t batch_index = 0;
    for (size_t begin = 0; begin < n_trials; begin += impl_->batch_capacity, ++batch_index) {
        const size_t batch_count = std::min(impl_->batch_capacity, n_trials - begin);
        const std::string batch_context = context + " trials [" + std::to_string(begin) + "," +
                                          std::to_string(begin + batch_count) + ")";
        if (injected_failure_batch == batch_index)
            throw std::runtime_error(batch_context + ": injected CUDA batch failure");
        const size_t mt_count = checked_mul(batch_count, 6, batch_context + " MT count");
        const size_t output_count =
            checked_mul(impl_->n_phases, batch_count, batch_context + " output count");
        impl_->timing.h2d_ms += measure_cuda(batch_context + " MT H2D", [&]() {
            copy_to_device(impl_->mt.data(), mt_host + begin * 6, mt_count,
                           batch_context + " MT H2D");
        });

        impl_->timing.kernel_ms += measure_cuda(batch_context + " kernel", [&]() {
            launch_xcorr_cuda(impl_->mt.data(), impl_->cc.data(), impl_->synamp.data(),
                              impl_->obs_norm2.data(), impl_->cc_max.data(), impl_->best_lag.data(),
                              impl_->need_energy ? impl_->energy.data() : nullptr,
                              static_cast<int>(impl_->n_phases), static_cast<int>(batch_count),
                              static_cast<int>(cc_rows), maxlag);
            if (impl_->need_amplitude)
                launch_waveform_cuda(impl_->mt.data(), impl_->gf.data(), impl_->amp_scale.data(),
                                     impl_->sign_scale.data(), static_cast<int>(impl_->n_phases),
                                     static_cast<int>(batch_count), static_cast<int>(n_samples));
        });

        impl_->timing.d2h_ms += measure_cuda(batch_context + " D2H", [&]() {
            copy_to_host(cc_max_host + begin * impl_->n_phases, impl_->cc_max.data(), output_count,
                         batch_context + " CC D2H");
            copy_to_host(best_lag_host + begin * impl_->n_phases, impl_->best_lag.data(),
                         output_count, batch_context + " lag D2H");
            if (impl_->need_energy)
                copy_to_host(energy_host + begin * impl_->n_phases, impl_->energy.data(),
                             output_count, batch_context + " energy D2H");
            if (impl_->need_amplitude) {
                copy_to_host(amp_scale_host + begin * impl_->n_phases, impl_->amp_scale.data(),
                             output_count, batch_context + " amplitude D2H");
                copy_to_host(sign_scale_host + begin * impl_->n_phases, impl_->sign_scale.data(),
                             output_count, batch_context + " sign D2H");
            }
        });
    }
}

} // namespace fm
