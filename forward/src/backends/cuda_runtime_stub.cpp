#include "backends/cuda_runtime.h"

#include <stdexcept>

namespace fm {

CudaProbeResult probe_cuda_device() {
    return {CudaProbeStatus::NotCompiled, "", "CUDA backend not compiled", 0.0};
}

struct CudaXcorrExecutor::Impl {};

CudaXcorrExecutor::CudaXcorrExecutor(size_t, size_t, size_t, size_t, std::optional<size_t>, bool,
                                     bool)
    : impl_(nullptr) {
    throw std::runtime_error("CUDA backend not compiled");
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
    return 0;
}

const CudaTimings &CudaXcorrExecutor::timings() const {
    static const CudaTimings timings;
    return timings;
}

void CudaXcorrExecutor::evaluate(const double *, const double *, const double *, const double *,
                                 const double *, double *, int32_t *, double *, double *, int8_t *,
                                 size_t, size_t, size_t, int, const std::string &) {
    throw std::runtime_error("CUDA backend not compiled");
}

} // namespace fm
