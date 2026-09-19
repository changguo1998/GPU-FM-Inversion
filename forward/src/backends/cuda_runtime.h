#ifndef FM_CUDA_RUNTIME_H
#define FM_CUDA_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace fm {

enum class CudaProbeStatus { Available, NoDevice, Error, NotCompiled };

struct CudaProbeResult {
    CudaProbeStatus status;
    std::string device_name;
    std::string error;
    double initialization_ms;
};

struct CudaTimings {
    double initialization_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
};

/// Probe logical CUDA device 0 without exposing CUDA headers to C++ callers.
CudaProbeResult probe_cuda_device();

class CudaXcorrExecutor {
  public:
    CudaXcorrExecutor(size_t n_phases, size_t max_cc_rows, size_t max_samples, size_t total_trials,
                      std::optional<size_t> batch_limit, bool need_energy, bool need_amplitude);
    ~CudaXcorrExecutor();
    CudaXcorrExecutor(const CudaXcorrExecutor &) = delete;
    CudaXcorrExecutor &operator=(const CudaXcorrExecutor &) = delete;
    CudaXcorrExecutor(CudaXcorrExecutor &&other) noexcept;
    CudaXcorrExecutor &operator=(CudaXcorrExecutor &&other) noexcept;

    size_t batch_capacity() const;
    const CudaTimings &timings() const;

    void evaluate(const double *mt, const double *cc, const double *synamp, const double *obs_norm2,
                  const double *gf, double *cc_max, int32_t *best_lag, double *energy,
                  double *amp_scale, int8_t *sign_scale, size_t n_trials, size_t cc_rows,
                  size_t n_samples, int maxlag, const std::string &context);

  private:
    struct Impl;
    Impl *impl_;
};

} // namespace fm

#endif // FM_CUDA_RUNTIME_H
