#include "backends/cuda_batch.h"

#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

template <typename F> void expect_throw(const std::string &name, F &&f) {
    try {
        f();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("expected failure: " + name);
}

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    constexpr size_t MIB = 1024 * 1024;
    const fm::CudaBatchPlan automatic =
        fm::plan_cuda_batch(1024 * MIB, 4096 * MIB, 9, 100000, std::nullopt);
    require(automatic.capacity > 0 && automatic.capacity <= 100000,
            "automatic capacity out of range");
    require(automatic.safety_margin == 256 * MIB, "wrong minimum safety margin");
    require(automatic.per_trial_bytes ==
                6 * sizeof(double) + 9 * sizeof(double) + 9 * sizeof(int32_t),
            "wrong per-trial byte count");

    const fm::CudaBatchPlan limited = fm::plan_cuda_batch(1024 * MIB, 4096 * MIB, 9, 1000, 7);
    require(limited.capacity == 7, "explicit limit not applied");
    require((1000 % limited.capacity) != 0, "fixture must exercise a residual batch");

    const fm::CudaBatchPlan one = fm::plan_cuda_batch(512 * MIB, 4096 * MIB, 9, 10, 1);
    require(one.capacity == 1, "batch=1 not preserved");

    expect_throw("zero limit", [&] { fm::plan_cuda_batch(512 * MIB, 4096 * MIB, 9, 10, 0); });
    expect_throw("no room after safety margin",
                 [&] { fm::plan_cuda_batch(128 * MIB, 4096 * MIB, 9, 10, std::nullopt); });
    expect_throw("phase byte overflow", [&] {
        fm::plan_cuda_batch(std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(),
                            std::numeric_limits<size_t>::max(), 10, std::nullopt);
    });

    const size_t many_phases = static_cast<size_t>(std::numeric_limits<int>::max()) / 2 + 1;
    const fm::CudaBatchPlan int_limited =
        fm::plan_cuda_batch(std::numeric_limits<size_t>::max() / 2,
                            std::numeric_limits<size_t>::max() / 2, many_phases, 10, std::nullopt);
    require(int_limited.capacity == 1, "INT_MAX work-item limit not applied");

    std::cout << "cuda_batch_test: PASS" << std::endl;
    return 0;
}
