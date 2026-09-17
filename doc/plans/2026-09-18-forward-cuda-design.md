# Forward CUDA 后端设计

**状态：** 独立审阅修订完成，待复审\
**范围：** `forward/` 计算阶段\
**目标硬件：** NVIDIA GeForce RTX 5060 Ti（16 GiB，CUDA architecture 120）\
**当前工具链：** Spack `cuda@13.2.1`，NVCC 13.2.78，CMake 3.28.3

## 1. 背景

当前管道为：

```text
input → preprocess → forward → assess → output → report
```

只有 `forward` 包含适合 GPU 的大规模独立计算。现有实现使用 C++17 +
OpenMP，以 `(phase, trial)` 为并行单元计算 XCorr 的 `cc_max` 和
`best_lag`。`forward/src/backends/device.h` 已有 CUDA dispatch 原型，但当前
CMake 只启用 CXX，`main.cpp` 显式调用 `Backend::OpenMP`，所有数组也都是
host 内存，因此尚无可运行的 GPU 路径。

现有原型还不能直接作为实施方案：它没有定义 CUDA 构建、显存生命周期、
host/device 传输、运行时后端选择、错误处理或 CPU/GPU 一致性验收。

## 2. 约束

- 保持 HDF5 schema 不变：GPU 与 CPU 写出相同的 `/intermediates`。
- 保持有符号 XCorr 语义：`misfit = 1 - cc_max`，不取绝对值。
- 保持 `best_lag` 的符号、单位和 tie-breaking 行为不变。
- 保留 OpenMP CPU 后端，便于回归和无 GPU 环境运行。
- 当前主流程只要求 XCorr P+S；Polarity/PSR 继续 deferred。
- 当前 synthetic 基线必须继续恢复相同 MT、depth 和 duration。
- HDF5 读写留在 CPU；不引入 GPU-aware HDF5。
- 输入、计算或写出失败必须返回非零状态，不得把未计算项当作零结果。
- CUDA 失败发生在 HDF5 commit 前，不得破坏已有 `/intermediates`。
- 不修改 input、preprocess、assess、output 或 report 的阶段职责。

## 3. 非目标

- 不实现多 GPU、MPI task distribution 或跨节点通信。
- 不把 Julia 阶段迁移到 GPU。
- 不接入新的 misfit operator。
- 不在首版重写 DataCache 的波形/GF reduction 算法。
- 不引入 Kokkos、Thrust 或新的第三方运行时依赖。
- 不以首版同时完成所有性能优化；先建立正确、可测的 CUDA 路径。
- 不承诺断电或底层 HDF5 文件损坏时的物理事务性；只保证阶段级逻辑恢复。

## 4. 决策清单

| ID | 设计问题 | 初始建议 | 状态 |
|-----|----------------------------|-----------------------------------------------|----------------------|
| D1 | 首版 GPU 加速范围 | 仅 XCorr trial evaluation | 已确认 |
| D2 | 构建产物与后端选择 | 单一 `forward`，CUDA 可选编译，运行时显式选择 | 已确认 |
| D3 | CUDA kernel 代码组织 | 共享 work-item 数学，CPU/CUDA 分别 launcher | 已确认 |
| D4 | 显存与传输生命周期 | combo 输入常驻当前轮，MT/输出按 batch 覆盖复用 | 已确认（独立审阅修订） |
| D5 | trial 分批与显存上限 | 自动计算 batch，支持显式限制 | 已确认（独立审阅修订） |
| D6 | CUDA stream、同步与错误处理 | 默认 stream，同步执行，每次 CUDA 调用检查 | 已确认 |
| D7 | 自动 fallback 行为 | 显式 CUDA 请求失败时不静默退回 CPU | 已确认 |
| D8 | 数值一致性验收 | `|Δcc_max| ≤ 1e-9`，`best_lag` 完全一致 | 已确认（独立审阅修订） |
| D9 | 性能验收 | 只记录分段耗时与加速比，不设性能门槛 | 已确认 |
| D10 | 工具链与目标架构 | CUDA 13.2.1；架构由 CMake 参数指定，当前用 120 | 已确认 |

以下各节保留推荐方案、备选方案和最终确认结果，作为实施依据。

### 4.1 独立审阅修订摘要

| ID | 审阅发现 | 修订结果 |
|----|----------------------------------------------------|-------------------------------------------------------------------|
| R1 | 泛型 `Device<B>` 可能把 CUDA 静默执行为 OpenMP | CPU/CUDA 使用显式非模板 launcher；禁止泛型 fallback |
| R2 | 全量 MT 常驻使分批无法约束全部显存 | MT 与输出都改为 batch buffer；combo reductions 在当前 combo 内复用 |
| R3 | 输入、combo 和 P/S shape 缺少 preflight | 增加统一 preflight、完成位图及 P/S 一致性约束 |
| R4 | 删除 `/intermediates` 后逐项写入不具备失败恢复能力 | 增加临时组、校验、flush、link swap 和启动恢复规则 |
| R5 | CLI、CPU-only/CUDA build 和 fallback 状态不完整 | 固化 CLI grammar、状态表、设备选择和 CMake 条件构建 |
| R6 | 浮点、规模计算、测试与计时契约不足 | 固化 FP flags、checked size、测试矩阵和计时方式 |

## 5. D1：首版 GPU 加速范围

**确认结果：** 首版只修改 `forward`，其中仅 XCorr trial evaluation 使用 GPU；
HDF5、trial 分组、DataCache reduction 和结果写回继续使用 CPU。

### 问题

`forward` 当前包含三类工作：

1. HDF5 读取和 trial 分组；
1. DataCache 在 CPU 上构造 XCorr reductions；
1. 对所有 `(phase, trial)` 计算 `cc_max` 与 `best_lag`。

首版 CUDA 后端应覆盖哪些工作？

### 推荐方案

只迁移第 3 类工作，即当前 `launch_xcorr_misfit` 对应的 trial evaluation。
HDF5、trial 分组和 DataCache reduction 保持 CPU 实现。

理由：

- trial evaluation 的并行规模最大，也是现有 backend 抽象对应的边界；
- 不改变输入 reductions、数据布局或 HDF5 契约；
- CPU/GPU 可直接消费相同输入并逐元素对比；
- 避免在未测量前同时重写两段算法，便于定位数值差异。

### 备选方案

同时把 DataCache reduction 搬到 GPU。可减少 CPU 计算，但需要额外设计波形/GF
传输、lag reduction 和缓存生命周期，扩大首版范围，也让错误定位更困难。

## 6. D2：构建产物与后端选择

**确认结果：** 只生成一个 `forward`。`FM_ENABLE_CUDA=OFF` 时保持 CPU-only；
`FM_ENABLE_CUDA=ON` 时同一程序包含 CPU/CUDA launcher，并通过
`--backend cpu|cuda|auto` 运行时选择。启动日志记录实际 backend 和 GPU。

CLI grammar 固定为：

```text
forward [--backend auto|cpu|cuda] [--cuda-batch-trials N] <database.h5> <status_N.h5>
```

原有两个位置参数的调用保持有效。环境变量 `FM_FORWARD_BACKEND` 和
`FM_CUDA_BATCH_TRIALS` 提供 driver 场景下的覆盖值；显式 CLI 参数优先于环境变量。

| 构建/请求 | 结果 |
|--------------------------------------------|-------------------------------------|
| CPU-only + `auto`/`cpu` | OpenMP |
| CPU-only + `cuda` | 明确报错“CUDA backend not compiled” |
| CUDA build + `cpu` | OpenMP |
| CUDA build + `cuda` | CUDA；无设备或初始化失败均报错 |
| CUDA build + `auto` + `cudaErrorNoDevice` | OpenMP |
| CUDA build + `auto` + 其他 CUDA 初始化错误 | 报错，不 fallback |

`--cuda-batch-trials` 必须是正整数，且只允许最终选择 CUDA 时使用。CUDA 使用
`CUDA_VISIBLE_DEVICES` 中的逻辑 device 0。首版若发现活跃的非 XCorr base operator，
显式/自动 CUDA 均明确拒绝，避免同一轮混合 CPU/GPU operator。

### 推荐方案

- CMake 增加 `FM_ENABLE_CUDA=ON|OFF`，默认保持 CPU-only 可构建。
- CUDA-enabled 构建仍生成单一 `forward` 可执行文件，同时包含 CPU 和 CUDA
  launcher。
- 运行时通过 `--backend cpu|cuda|auto` 选择，CUDA-enabled 构建默认 `auto`。
- 启动日志必须记录最终使用的 backend 和 GPU 名称。

### 备选方案

分别生成 `forward_cpu`、`forward_cuda`。实现简单，但 driver 和部署需要选择两个
文件，容易让两条入口漂移。

## 7. D3：CUDA kernel 代码组织

**确认结果：** XCorr 的单个 `(phase, trial)` 数学实现只保留一份；OpenMP 和
CUDA 分别提供调度器调用该实现。CUDA launcher 独立放在 `.cu` 中，不让 NVCC
编译整个 `main.cpp`。

独立审阅后补充：launcher 使用 CXX 可见的普通函数接口，不从普通 C++ translation
unit 实例化 `Device<Backend::CUDA>`。现有“任意 `Backend B` 默认走 OpenMP”的泛型
实现必须删除或改为未定义 primary template，只允许显式 OpenMP 实现；CUDA 入口只
能链接到 `.cu` 中的实现。CPU-only build 提供“未编译 CUDA”的查询结果，不提供
伪 CUDA launcher。

### 推荐方案

共享单个 host/device work-item 数学实现，CPU 和 CUDA 只分别负责调度：

```text
XCorrWorkItem::operator()(index)  ← 唯一数值实现
OpenMP launcher                   ← omp parallel for
CUDA launcher                     ← __global__ kernel
```

CUDA launcher 放入 `.cu` 文件。避免把包含 HDF5 和 STL 控制流的整个 `main.cpp`
交给 NVCC，也避免依赖 extended-lambda 才能编译现有模板。

### 备选方案

将 `main.cpp` 整体作为 CUDA 编译，并继续使用 `Device<Backend>::parallel_for`
lambda。改动表面较少，但 host/device lambda 标注、NVCC 编译边界和 HDF5 host
代码会耦合在一起。

## 8. D4：显存与传输生命周期

**确认结果：** frequency/depth/duration combo 的 reductions 在处理当前 combo 时
保留在显存；MT、`cc_max` 和 `best_lag` 使用固定容量的 batch buffer。完整输出保留在
host。所有 device buffer 按运行期间最大需求分配一次并覆盖复用，不在 batch/combo
之间重复 `cudaMalloc/cudaFree`。

### 推荐方案

搜索维度分成两类：

- `strike/dip/rake` 决定每个 trial 的 MT；
- `frequency/depth/duration` 决定该 trial 使用哪组 GF/XCorr reductions。

每个 trial 只属于一个 combo，因此全量 MT 一次传输与逐 batch 传输的总字节数相同；
全量常驻反而使显存无法由 batch size 约束。整个 forward 采用：

```text
启动：在 host 分配完整 cc_max/best_lag 输出；一次分配可复用 device buffers

每个 (frequency, depth, duration) combo：
    → 覆盖写入可复用的 cc/synamp/obs_norm2 buffer
    → host 按 trial_indices 打包当前 batch 的 MT
    → 覆盖写入可复用的 MT batch buffer
    → kernel 写入可复用的 batch 输出 buffer
    → 将当前 batch 输出复制回 host 全局数组

结束：host 全局数组写入 HDF5
```

所有 device buffer 由小型 RAII 类型管理。combo buffer 按运行期间所需最大容量
分配一次，或只增不减；后续 combo 只覆盖内容，不重复 `cudaMalloc/cudaFree`。

不把所有 combo 的 GF/reductions 或 MT 同时常驻显存，因为每项只消费一次；常驻的
是整个运行期间反复覆盖使用的 buffer，以及处理当前 combo 所需的 reductions。

### 备选方案

把全部 DataCache entries 或全部 MT 常驻 GPU。不能减少总 MT 传输量，显存占用却随
搜索空间增长，且当前每项只使用一次，首版收益有限。

## 9. D5：trial 分批与显存上限

**确认结果：** 首版支持 trial 分批。程序根据可用显存自动计算 batch size，同时
允许通过 `--cuda-batch-trials N` 显式设置上限；该参数也用于测试强制走多 batch
路径。

### 推荐方案

选择逻辑 device 0 并完成 CUDA context 初始化后，先按所有 host cache entries 的最大
shape 分配 combo input buffers，再调用 `cudaMemGetInfo`。保留
`max(256 MiB, total_device_memory × 5%)` 安全余量，剩余空间按每个 trial 的 batch
输入/输出大小计算容量：

```text
per_trial_bytes = 6 × sizeof(Float64)                  # MT
                + N_phases × sizeof(Float64)          # cc_max
                + N_phases × sizeof(Int32)            # best_lag
```

自动值与 `--cuda-batch-trials N` 上限取较小者。batch buffer 只按最终容量分配一次，
所有 combo 重复使用。所有元素数、字节数和乘法使用 checked `size_t`，防止 unsigned
下溢或溢出；同时限制 `N_phases × batch_size ≤ INT_MAX`，首版 kernel 保持 32-bit
work-item index。若连一个 trial 都无法容纳，则报告 combo、batch、安全余量和可用
显存后退出，不静默切换 CPU。

### 备选方案

固定写死 batch size。实现稍简单，但无法适配不同 GPU、phase 数量和输入规模，
也可能浪费显存或在小显存设备上失败。

## 10. D6：stream、同步与错误处理

**确认结果：** 首版只使用 CUDA default stream，batch 顺序执行；不引入 pinned
host memory、双缓冲或传输/计算重叠。所有 CUDA 调用和 kernel launch 必须检查错误，
错误上下文包含 combo、batch 和数组 shape。

### 推荐方案

- 使用 CUDA default stream；首版不重叠 H2D、kernel 和 D2H。
- 所有 CUDA runtime 调用经统一 `CUDA_CHECK` 检查。
- kernel launch 后检查 `cudaGetLastError()`。
- D2H 前显式同步，确保异步 kernel 错误在当前 combo 内报告。
- 错误信息包含操作名、combo key、数组 shape 和 CUDA error string。
- 任何 CUDA 错误都发生在 HDF5 commit 前；host 中未完成的输出不得进入写盘路径。

### 备选方案

使用多个 stream 和 pinned memory 做流水线。可能提高吞吐，但会让资源生命周期和
错误归属复杂化，应在 profiler 证明传输是瓶颈后再做。

## 11. D7：自动 fallback 行为

**确认结果：** CUDA-enabled 构建默认使用 `auto`。`auto` 只允许在启动检测阶段因无
CUDA device 选择 CPU；一旦开始 CUDA 执行，中途错误直接失败。显式 `cuda` 的任何
初始化或执行错误均直接失败，显式 `cpu` 始终使用 OpenMP。

### 推荐方案

- `cpu`：始终使用 OpenMP。
- `cuda`：没有 CUDA device 或发生 CUDA 初始化错误时直接失败。
- `auto`：启动时没有 CUDA device 才选择 CPU；开始执行后出现 CUDA 错误不得
  fallback，以免掩盖实现缺陷或产生部分 CPU/部分 GPU 结果。

### 备选方案

任何 CUDA 错误都自动退回 CPU。用户体验更宽松，但可能掩盖错误，且 combo 中途
失败时需要重新初始化全部输出。

## 12. D8：数值一致性验收

**确认结果：** CPU/GPU `cc_max` 绝对误差不超过 `1e-9`；`best_lag`、NaN/退化
行为及 tie-breaking 完全一致。synthetic 最终 MT、depth、duration 一致，并同时
覆盖自动 batch 与强制多 batch 路径。

浮点执行契约：CPU work-item 使用 `-ffp-contract=off`；CUDA work-item 使用
`--fmad=false`，禁止 `--use_fast_math`。lag 始终从小到大串行扫描，以严格 `>` 更新
最优值，因此精确并列选择首个有效 lag。若 CPU/GPU 仍出现 lag 差异，实施暂停并分析
输入，不通过放宽 lag 验收或引入未评审的 epsilon comparator 规避。

### 推荐方案

- 同一输入分别执行 CPU 和 GPU backend。
- `cc_max`：所有有限值绝对误差不超过 `1e-9`。
- `best_lag`：逐元素完全一致。
- NaN mask、Inf 拒绝和退化输入行为完全一致。
- GPU 结果继续通过独立 Julia reference 测试。
- synthetic e2e 的最佳 MT、depth、duration 与 CPU 基线一致。

若极少数近似并列 lag 仍因平台舍入产生不同 argmax，视为验收失败并暂停实施；不得
通过放宽 `best_lag` 验收掩盖。

## 13. D9：性能验收

**确认结果：** 只记录 CPU/GPU 分段耗时、总耗时和实际加速比，不把任何性能数字
作为首版验收门槛。

### 推荐方案

分别报告：

- HDF5/DataCache CPU 时间；
- H2D 时间；
- kernel 时间；
- D2H 时间；
- forward 总时间。

forward 总时间、HDF5/DataCache 和同步 H2D/D2H 使用 host monotonic clock；kernel
使用 CUDA events。CUDA context 初始化单独记录，并包含在实际 forward 总时间中。
计时所需同步与 D6 的错误检查同步点合并，不增加额外流水线语义。

首版完成标准是正确性和可重复测量，不预设加速倍数，也不要求 GPU 必须快于 CPU。
获得 synthetic 和至少一个真实规模数据的测量后，再决定是否设性能目标及优化位置。

## 14. D10：工具链与目标架构

**确认结果：** 使用现有 Spack `cuda@13.2.1`。CUDA 构建显式传入
`FM_ENABLE_CUDA=ON` 和 `CMAKE_CUDA_ARCHITECTURES=120`；架构不写死在源码，
不使用 `native`，首版不生成多架构 fat binary。CPU-only 构建不依赖 NVCC。

CMake 最低版本提升至 3.18。`FM_ENABLE_CUDA=ON` 时才调用
`enable_language(CUDA)`、`find_package(CUDAToolkit REQUIRED)`，要求
`CMAKE_CUDA_ARCHITECTURES` 非空，设置 CUDA C++17 并链接 `CUDA::cudart`；OFF 时
不得探测 CUDA，也不得引用 CUDA header 或 symbol。

当前机器的可复现构建入口：

```bash
source ~/.spack/share/spack/setup-env.sh
spack load cuda@13.2.1
cmake -S forward -B forward/build-cuda \
    -DFM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build forward/build-cuda -j8
```

### 推荐方案

- 使用现有 Spack `cuda@13.2.1` 和 GCC host compiler。
- CMake 启用 CUDA language，并要求调用方通过
  `CMAKE_CUDA_ARCHITECTURES` 指定架构。
- 当前机器构建参数为 `-DCMAKE_CUDA_ARCHITECTURES=120`。
- 不把 `120` 写死在 kernel 源码；其他 GPU 可重新配置构建目录。
- CUDA-disabled 构建不需要 NVCC，继续支持原 OpenMP 环境。

### 备选方案

默认使用 `native`。本机配置方便，但构建产物不可预期地依赖构建机，不利于复现。

## 15. 输入 preflight 与计算完整性

backend 选择和任何输出分配前，统一验证：

- `N_trials > 0`，六个 trial index 数组长度都严格等于 `N_trials`；
- strike/dip/rake/depth/frequency/duration 全部为合法的 1-based paraspace index；
- 每个 trial 恰好归属一个 `(frequency, depth, duration)` combo；
- 每个 combo 的 DataCache entry 存在且所有数组 shape 与声明一致；
- 所有输入有限性符合 kernel 契约，Inf 直接拒绝，允许的 NaN 位置显式定义；
- 所有 host/device 元素数和字节数乘法使用 checked `size_t`。

删除“读取/计算 combo 失败后 `continue`”行为；任何缺失或 shape 错误直接失败。host
维护 `computed[N_trials]` 完成位图，每个 batch 回传后标记对应 trial；HDF5 commit 前
断言每个 trial 恰好完成一次，不能用初始化零值代表未计算。

当前 P/S 共用一个 XCorr cache entry 和相同 `cc_rows/maxlag`。首版 preflight 要求所有
活跃 XCorr P/S 数据具有相同 dt、band、`max_lag_periods`、window length、最终
`cc_rows` 和 `maxlag`；不满足时明确失败。未来如需不同 P/S 配置，再按模块拆分 entry
和 launcher，不在首版隐式兼容。

## 16. HDF5 commit 与失败恢复

所有 preflight、DataCache 和 CPU/GPU 计算先在 host 完成，期间不修改现有
`/intermediates`。写出采用以下逻辑事务：

1. 清理上次未完成且可安全判定的临时 link；
1. 完整写入 `/intermediates.__tmp__`；
1. 校验所有 dataset 的名称、类型、shape 和 trial 完成位图；
1. `H5Fflush` 成功后，将旧 `/intermediates` 移到 `/intermediates.__backup__`；
1. 将临时组移动为 `/intermediates`，再次 flush；
1. 删除 backup 并最终 flush。

启动恢复规则：final 存在时保留 final、清理残留 tmp/backup；final 不存在但 backup
存在时先恢复 backup，再清理 tmp；只有 tmp 存在表示 commit 未开始，直接删除 tmp。
任何 HDF5 调用失败都必须传播为非零退出，不得静默忽略。

该方案保证 stage 可检测失败后的逻辑恢复，不承诺断电导致底层 HDF5 文件损坏时的
物理事务性。若未来要求掉电级原子性，应改为临时 status 文件加文件级 atomic rename。

## 17. 构建与测试矩阵

必须覆盖：

- clean CPU-only configure/build，运行默认、`auto` 和显式 `cpu`；
- CPU-only 显式 `cuda` 失败，且错误信息明确；
- clean CUDA configure/build，同一 binary 分别强制 `cpu`、`cuda`；
- CPU/GPU 使用两份独立 status 文件，避免后运行覆盖前结果；
- 自动 batch，以及强制 batch size 为 1、质数、不能整除 combo 的情况；
- 多个大小不同的 freq/depth/duration combo，验证 buffer 覆盖无残留；
- 非连续、乱序 trial index 与 XCorr P+S；
- 精确 tie、近 tie、全负相关、零 norm、NaN、Inf 和退化 synamp；
- 无 device 时 `auto` 回退，显式 `cuda` 失败；其他初始化错误不得 fallback；
- preflight 错误、combo 缺失和 HDF5 故障注入不破坏旧 `/intermediates`；
- synthetic CUDA e2e 日志明确显示实际 backend 为 CUDA；
- `compute-sanitizer` 在强制多 batch 路径下无错误。

GPU 测试使用显式 opt-in；一旦在声明有 GPU 的验证环境启用，不允许把失败转换为
skip。CPU/GPU 对照继续使用独立 Julia reference，而不是只互相比较。

## 18. 预期代码边界

逐项确认后，预计只修改或新增：

```text
forward/CMakeLists.txt
forward/src/backends/device.h
forward/src/backends/cuda_runtime.{h,cu}
forward/src/kernels/xcorr_kernel.h
forward/src/kernels/xcorr_kernel.cu
forward/src/main.cpp
forward/src/data_cache.{h,cpp}
forward/src/hdf5_io.{h,cpp}
forward/AGENTS.md
doc/design.md
doc/modules/misfit_kernel.md
doc/modules/mt_utils.md
doc/stages/forward.md
tests/stages/forward_test.jl
tests/stages/e2e_test.jl
```

文件名可在实施时微调；不预先创建未使用的抽象。

## 19. 验收门槛

1. CPU-only 构建和现有 forward 测试继续通过。
1. CUDA-enabled 构建成功，明确记录 backend 和设备。
1. CPU/GPU 对同一 status 文件的 `cc_max` 与 `best_lag` 满足 D8。
1. GPU synthetic e2e 恢复相同最佳解。
1. `compute-sanitizer` 不报告越界、未初始化访问或泄漏。
1. 重复运行 forward 仍保持 HDF5 输出幂等。
1. 任意 preflight、CUDA 或 HDF5 commit 失败后，旧 `/intermediates` 仍可恢复。
1. D17 的构建、backend、batch、边界输入和故障矩阵全部通过。
1. 文档记录实际构建、运行和回归命令。

## 20. 待更新的旧文档

实施时同步修正以下已过期描述：

- `doc/modules/misfit_kernel.md` 的 XCorrS-only 表述；当前 synthetic 使用 XCorr P+S。
- `doc/modules/mt_utils.md` 将 C++ MT 工具称为“历史参考”，但当前 `main.cpp` 实际
  使用 `sdr_to_mt`。
- `doc/design.md` 和 `forward/AGENTS.md` 当前只记录 OpenMP CPU backend。
