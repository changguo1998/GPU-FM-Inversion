# Forward CUDA 后端实施任务

> 本文件是 `doc/plans/2026-09-18-forward-cuda-design.md` 的可执行任务清单。
> 每个 Task 应独立完成、验证并审查；除非用户明确要求，不执行建议 commit。

**目标：** 为 `forward` 增加可选 CUDA XCorr 后端，同时保留 OpenMP CPU 后端、现有
HDF5 schema 和有符号 XCorr 数学语义。

**实施策略：** 先固定 CPU 行为并补齐输入/写出安全性，再抽取 CPU/CUDA 共用的
XCorr work-item，随后加入条件 CUDA 构建、显存复用和分批调度。最后用独立 Julia
reference、synthetic 全流程和 compute-sanitizer 验收。

**工具链：** C++17、OpenMP、HDF5、CMake ≥ 3.18、CUDA 13.2.1、Julia/HDF5.jl。

## 全局约束

- 只加速 `forward` 的 XCorr trial evaluation；HDF5、DataCache reduction 和 Julia
  阶段继续使用 CPU。
- 不接入 Polarity/PSR，不修改 HDF5 schema，不实现多 GPU、异步流水线或 pinned
  memory。
- 保持 `misfit = 1 - cc_max`，不取绝对值；lag 从小到大扫描，以严格 `>` 更新。
- CPU 使用 `-ffp-contract=off`；CUDA 使用 `--fmad=false`，禁止 fast-math。
- CPU/GPU 验收：有限 `cc_max` 的绝对误差 `≤ 1e-9`，`best_lag` 完全一致。
- CPU-only 构建不得探测、包含或链接 CUDA；CUDA 构建架构由调用方显式指定。
- 所有规模与字节数计算使用 checked `size_t`；CUDA work-item 数不超过 `INT_MAX`。
- 所有计算成功并通过完成位图校验后，才允许提交 `/intermediates`。
- 代码、注释使用英文；文档与沟通使用中文；修改后运行 `bash format.sh`。
- GPU 命令需要在可见 RTX 5060 Ti 的终端执行；启用 GPU 测试后，失败不得转为 skip。

## 文件边界

| 文件 | 预期动作 | 职责 |
|----------------------------------------------|----------|--------------------------------------------------------|
| `forward/CMakeLists.txt` | 修改 | CPU-only/CUDA 条件构建、FP flags、测试目标 |
| `forward/src/backends/device.h` | 修改 | 删除泛型 CUDA→OpenMP 静默 fallback，仅保留明确 CPU 调度 |
| `forward/src/backends/cuda_runtime.h` | 新增 | CXX 可见的 CUDA capability、batch 和 launcher 接口 |
| `forward/src/backends/cuda_runtime_stub.cpp` | 新增 | CPU-only 构建的“CUDA 未编译”查询实现 |
| `forward/src/backends/cuda_runtime.cu` | 新增 | CUDA runtime、RAII buffer、batch planner、计时 |
| `forward/src/kernels/xcorr_kernel.h` | 修改 | 唯一 host/device XCorr work-item 数学与 CPU launcher |
| `forward/src/kernels/xcorr_kernel.cu` | 新增 | CUDA kernel 与 launcher |
| `forward/src/main.cpp` | 修改 | CLI、backend 状态机、preflight、combo/batch 编排、日志 |
| `forward/src/data_cache.{h,cpp}` | 修改 | cache shape/有限性校验、P/S invariant、失败传播 |
| `forward/src/hdf5_io.{h,cpp}` | 修改 | checked HDF5 操作、flush、link move、事务恢复与提交 |
| `forward/tests/*.cpp` | 按需新增 | checked size、preflight、HDF5 commit 的无 GPU 单测 |
| `tests/stages/test_util.jl` | 修改 | 可选择 forward 可执行文件与 backend 的测试入口 |
| `tests/stages/forward_test.jl` | 修改 | CPU reference、backend、batch、边界与恢复测试 |
| `tests/stages/e2e_test.jl` | 修改 | CUDA synthetic 全流程与实际 backend 日志断言 |
| `forward/AGENTS.md`、`doc/design.md` | 修改 | 构建和架构说明 |
| `doc/modules/{misfit_kernel,mt_utils}.md` | 修改 | XCorr P+S、共享公式、MT 实际用途 |
| `doc/stages/forward.md` | 修改 | CLI、fallback、batch、事务、验证命令 |

文件名可在实现时按现有代码边界微调，但不得预先创建没有实际调用者的抽象。

______________________________________________________________________

## Task 1：固定 CPU 基线与测试入口

**Files:**

- Modify: `tests/stages/test_util.jl`
- Modify: `tests/stages/forward_test.jl`
- Modify: `tests/stages/assess_test.jl`
- Modify: `tests/stages/output_test.jl`

**Interfaces:**

- `forward_executable()`：优先读取 `FM_FORWARD_EXE`，否则返回现有
  `forward/build/forward`。

- `run_forward(db, status; backend=nothing, batch_trials=nothing)`：统一构造命令；未指定
  参数时保持旧调用行为。

- [ ] **Step 1：记录未修改代码的 CPU 基线**

  构建 `forward/build-cpu`，运行现有 `forward_test.jl`，记录 reference 最大 CC 误差、
  lag 差异和 synthetic 最优解。基线失败时先停下，不把既有失败带入 CUDA 工作。

- [ ] **Step 2：参数化测试可执行文件**

  在测试辅助函数集中解析 `FM_FORWARD_EXE`；替换 stage 测试中硬编码的
  `forward/build/forward`，不改变默认路径。

- [ ] **Step 3：验证两套 build 目录互不覆盖**

  分别让同一测试指向 `build-cpu/forward` 和复制的第二个 CPU binary，证明测试选择
  来自环境变量，而非固定路径。

**Verify:**

```bash
cmake -S forward -B forward/build-cpu
cmake --build forward/build-cpu -j8
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    julia --project=. tests/stages/forward_test.jl
```

**建议 commit：** `test: 参数化 forward 阶段测试入口`

______________________________________________________________________

## Task 2：提取唯一 XCorr work-item，保持 CPU 结果不变

**Files:**

- Modify: `forward/src/backends/device.h`
- Modify: `forward/src/kernels/xcorr_kernel.h`
- Modify: `forward/src/main.cpp`
- Modify: `forward/CMakeLists.txt`
- Modify: `tests/stages/forward_test.jl`

**Interfaces:**

- `xcorr_work_item(...)`：单个 `(phase, trial)` 的唯一数值实现，可由 host/device 调用。

- `launch_xcorr_openmp(...)`：显式 OpenMP launcher；不再使用
  `launch_xcorr_misfit<Backend::OpenMP>`。

- 泛型 `Device<B>` primary template 不得默认执行 OpenMP；CUDA 不能在普通 C++ translation
  unit 中伪装为 CPU。

- [ ] **Step 1：先补数学边界失败测试**

  在独立 Julia reference 对照中加入精确 tie、近 tie、全负相关、零 obs norm、退化
  synamp 和允许的 NaN 行为。明确断言 tie 选择第一个有效 lag、有符号负相关不取绝对值。

- [ ] **Step 2：提取 work-item 与显式 CPU launcher**

  只移动数学，不改数据布局、循环顺序或输出语义。删除 `DefaultDevice` 和泛型
  fallback；普通 C++ 代码只能调用 `launch_xcorr_openmp`。

- [ ] **Step 3：锁定 CPU 浮点契约**

  给 forward C++ target 增加 `-ffp-contract=off`；不启用 fast-math。重新运行 reference
  测试，要求结果与 Task 1 基线一致。

**Verify:**

```bash
cmake --build forward/build-cpu -j8
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    julia --project=. tests/stages/forward_test.jl
```

**建议 commit：** `refactor: 提取共享 XCorr work-item`

______________________________________________________________________

## Task 3：实现统一输入 preflight 与计算完整性检查

**Files:**

- Modify: `forward/src/main.cpp`
- Modify: `forward/src/data_cache.h`
- Modify: `forward/src/data_cache.cpp`
- Add: `forward/tests/preflight_test.cpp`
- Modify: `forward/CMakeLists.txt`
- Modify: `tests/stages/forward_test.jl`

**Interfaces:**

- `checked_add` / `checked_mul`：溢出时报带变量名的异常。

- `validate_forward_input(...)`：在 backend 选择和输出分配前完成全部结构校验。

- `computed[N_trials]`：每个 trial 回写后从 0 变为 1；重复计算或未计算均失败。

- [ ] **Step 1：写失败 fixture**

  覆盖：`N_trials == 0`、六个 index 数组长度不一致、六类 index 越界、缺失 combo、
  shape 不符、Inf、checked-size 溢出、同一 trial 重复/遗漏，以及 P/S 的 dt、band、
  `max_lag_periods`、window length、`cc_rows` 或 `maxlag` 不一致。

- [ ] **Step 2：实现 paraspace 与 trial 校验**

  完整读取 strike/dip/rake/depth/frequency/duration 轴；删除 `axis_val` 越界返回 0.0 的
  行为。验证六个 trial 数组长度严格等于 `N_trials`，所有 index 为合法 1-based 值。

- [ ] **Step 3：实现 combo/cache 校验**

  DataCache 加载失败、entry 缺失或 shape 不合法必须抛错；删除 main 中
  catch-and-continue 和 invalid-entry continue。固定首版 P/S 共用 reduction 的 invariant。

- [ ] **Step 4：加入完成位图**

  combo 分组后先验证每个 trial 恰好属于一组；每个 batch 回写时检查并标记。HDF5
  commit 前要求所有值恰好为 1，禁止以初始化零值代表未计算结果。

**Verify:**

```bash
ctest --test-dir forward/build-cpu --output-on-failure
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    julia --project=. tests/stages/forward_test.jl
```

**建议 commit：** `fix: 严格校验 forward 输入与计算完整性`

______________________________________________________________________

## Task 4：实现可恢复的 HDF5 intermediates 提交

**Files:**

- Modify: `forward/src/hdf5_io.h`
- Modify: `forward/src/hdf5_io.cpp`
- Modify: `forward/src/main.cpp`
- Add: `forward/tests/hdf5_commit_test.cpp`
- Modify: `forward/CMakeLists.txt`
- Modify: `tests/stages/forward_test.jl`

**Interfaces:**

- `recover_intermediates()`：按 final/tmp/backup 组合恢复到唯一 final。

- `commit_intermediates(writer, validator)`：写 tmp、校验、flush、link swap、清 backup。

- HDF5 link delete/move/flush 失败必须抛错；不得再静默忽略返回码。

- [ ] **Step 1：写恢复状态测试**

  构造 final-only、final+tmp、final+backup、backup-only、tmp-only 等状态，逐项验证设计
  §16 的恢复规则。旧 final 内容使用唯一 marker，确保没有被错误覆盖。

- [ ] **Step 2：写 commit 故障注入测试**

  C++ 测试通过内部 fault point 在 tmp 写完、首次 flush、old→backup、tmp→final、最终
  flush 等边界抛错；重新打开文件并恢复后，promotion 前必须保留旧 final，promotion 后
  必须得到完整、已校验的新 final 或回滚后的旧 final，绝不接受部分结果。fault point
  仅供测试调用，不增加用户 CLI 或环境变量。

- [ ] **Step 3：实现 checked HDF5 操作与恢复**

  增加 link move、flush 和 dataset metadata 查询；所有 close/delete/move/flush 传播错误。
  启动打开 status 后先恢复残留事务状态。

- [ ] **Step 4：替换 delete-and-recreate 写法**

  所有模块完整写入 `/intermediates.__tmp__`，校验名称、类型和 shape 后统一提交。
  计算/preflight 失败不得创建或改动 tmp/final/backup。

**Verify:**

```bash
ctest --test-dir forward/build-cpu --output-on-failure
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    julia --project=. tests/stages/forward_test.jl
```

**建议 commit：** `fix: 原子化提交 forward intermediates`

______________________________________________________________________

## Task 5：加入 CLI、backend 状态机与条件 CUDA 构建

**Files:**

- Modify: `forward/CMakeLists.txt`
- Modify: `forward/src/main.cpp`
- Add: `forward/src/backends/cuda_runtime.h`
- Add: `forward/src/backends/cuda_runtime_stub.cpp`
- Add: `forward/src/backends/cuda_runtime.cu`
- Modify: `tests/stages/forward_test.jl`

**Interfaces:**

```text
forward [--backend auto|cpu|cuda] [--cuda-batch-trials N] <database.h5> <status_N.h5>
```

- 环境变量：`FM_FORWARD_BACKEND`、`FM_CUDA_BATCH_TRIALS`；CLI 优先。

- `FM_ENABLE_CUDA=OFF`：只编译 CPU stub，不引用 CUDA header/symbol。

- `FM_ENABLE_CUDA=ON`：同一 binary 包含 CPU/CUDA，默认 backend 为 `auto`。

- [ ] **Step 1：写 CLI/backend 矩阵失败测试**

  覆盖旧双位置参数调用、未知选项、缺失参数、非法 backend、batch 为 0/负数/非整数、
  CLI 覆盖环境变量、CPU-only 显式 cuda 报“CUDA backend not compiled”。

- [ ] **Step 2：实现严格 CLI parser**

  保持原调用有效。`--cuda-batch-trials` 只有最终选择 CUDA 时合法；错误返回非零并输出
  完整 usage，不修改 status 文件。

- [ ] **Step 3：实现条件构建**

  CMake 最低版本升至 3.18，增加默认 OFF 的 `FM_ENABLE_CUDA`。仅 ON 时
  `enable_language(CUDA)`、查找 CUDAToolkit、要求非空 CUDA architecture、链接 cudart；
  CUDA C++17，设置 `--fmad=false`。OFF 时编译不包含 CUDA header/symbol 的 stub。

- [ ] **Step 4：实现启动期 backend 状态机**

  使用逻辑 device 0。显式 cuda 的无设备/初始化错误均失败；auto 仅对
  `cudaErrorNoDevice` 回退 CPU，其他错误失败。一旦开始 CUDA 计算，不允许中途回退。
  活跃 base operator 非 XCorr 时，CUDA/auto 明确拒绝。

- [ ] **Step 5：记录实际选择**

  stdout 记录 requested backend、selected backend；CUDA 路径额外记录逻辑 device 和
  GPU 名称。后续 e2e 以该日志断言确实运行 CUDA。

**Verify:**

```bash
cmake -S forward -B forward/build-cpu -DFM_ENABLE_CUDA=OFF
cmake --build forward/build-cpu -j8

source ~/.spack/share/spack/setup-env.sh
spack load cuda@13.2.1
cmake -S forward -B forward/build-cuda \
    -DFM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build forward/build-cuda -j8
```

**建议 commit：** `feat: 增加 forward CUDA 构建与后端选择`

______________________________________________________________________

## Task 6：实现 CUDA 资源管理与 batch planner

**Files:**

- Modify: `forward/src/backends/cuda_runtime.h`
- Modify: `forward/src/backends/cuda_runtime.cu`
- Modify: `forward/src/main.cpp`
- Add: `forward/tests/cuda_batch_test.cpp`
- Modify: `forward/CMakeLists.txt`

**Interfaces:**

- move-only `CudaBuffer<T>`：容量只增不减，析构释放；所有 runtime 调用 checked。

- `plan_cuda_batch(...)`：根据 combo 最大容量、可用显存、安全余量和用户上限返回正容量。

- `CudaTimings`：context、H2D、kernel、D2H 的累计耗时。

- [ ] **Step 1：写纯 CPU batch planner 单测**

  覆盖自动容量、显式上限、batch=1、质数上限、残 batch、恰好一项、连一个 trial 都
  放不下、`N_phases × batch > INT_MAX`、所有 size 乘加溢出。planner 的纯算法放在 CXX
  可编译边界，不依赖 CUDA runtime 或 GPU。

- [ ] **Step 2：实现 RAII buffer 与错误上下文**

  为 combo 的 cc/synamp/obs_norm2 和 batch 的 MT/cc_max/best_lag 分配可复用 buffer。
  后续 combo 只覆盖或扩容，不在 batch 间 cudaMalloc/cudaFree。错误包含操作、combo、
  batch 范围和 shape。

- [ ] **Step 3：实现自动 batch 计算**

  先为最大 combo input 预留空间，再调用 `cudaMemGetInfo`。安全余量为
  `max(256 MiB, total × 5%)`；每 trial 字节数严格按设计 §9 计算。自动值与显式上限
  取较小者。

- [ ] **Step 4：实现计时基础设施**

  context/H2D/D2H/总时间使用 host monotonic clock；kernel 使用 CUDA events。复用后续
  已有同步点，不引入额外 stream 或并行语义。

**Verify:**

```bash
ctest --test-dir forward/build-cpu --output-on-failure
ctest --test-dir forward/build-cuda --output-on-failure
```

**建议 commit：** `feat: 增加 CUDA 显存复用与分批规划`

______________________________________________________________________

## Task 7：实现 CUDA XCorr launcher 并接入 combo 循环

**Files:**

- Add: `forward/src/kernels/xcorr_kernel.cu`
- Modify: `forward/src/kernels/xcorr_kernel.h`
- Modify: `forward/src/backends/cuda_runtime.h`
- Modify: `forward/src/backends/cuda_runtime.cu`
- Modify: `forward/src/main.cpp`
- Modify: `forward/CMakeLists.txt`
- Modify: `tests/stages/forward_test.jl`

**Interfaces:**

- `launch_xcorr_cuda(...)`：普通 CXX 可调用、实现仅位于 `.cu`；输入输出都是明确 flat
  pointer、shape 和 batch 范围。

- 单 default stream，严格顺序：combo H2D → batch MT H2D → kernel → sync → batch D2H。

- [ ] **Step 1：写首个 CPU/GPU parity 测试**

  从同一输入复制两份 status，分别强制 `--backend cpu` 与 `--backend cuda`。在实现前
  GPU 用例应因 launcher 尚未接入而失败，不能以 skip 通过。

- [ ] **Step 2：实现 CUDA kernel**

  每个 thread 处理一个 `(phase, trial)`，只调用共享 `xcorr_work_item`。kernel launch 后
  检查 `cudaGetLastError()`，D2H 前同步并检查异步错误。保持 32-bit work-item index。

- [ ] **Step 3：接入 combo/batch 数据流**

  当前 combo reductions 只上传一次；按原始、可乱序 `trial_indices` 打包 batch MT，
  回传后 scatter 到完整 host 输出并更新完成位图。完整输出留在 host，最终统一 HDF5
  commit。

- [ ] **Step 4：验证 buffer 覆盖**

  构造多个大小和 shape 不同的 freq/depth/duration combo，后一个 combo 必须完全覆盖
  前一个数据，结果不能出现显存残留。

**Verify:**

```bash
FM_FORWARD_EXE="$PWD/forward/build-cuda/forward" \
    julia --project=. tests/stages/forward_test.jl
```

Expected：日志为 `selected backend: cuda`，CC 误差 `≤ 1e-9`，lag 完全一致。

**建议 commit：** `feat: 实现 CUDA XCorr trial evaluation`

______________________________________________________________________

## Task 8：完成 batch、乱序与数值边界测试矩阵

**Files:**

- Modify: `tests/stages/forward_test.jl`

- Modify: `tests/stages/test_util.jl`

- [ ] **Step 1：分离 CPU/GPU status**

  每个 backend 使用独立 status 副本，禁止后运行覆盖前一结果；比较前同时验证两份
  `/intermediates` schema、类型和 shape。

- [ ] **Step 2：覆盖 batch 分割**

  对相同 fixture 运行自动 batch、1、质数和不能整除 combo 的 batch。每一种都对独立
  Julia reference，而不是只和 CPU 互比。

- [ ] **Step 3：覆盖 trial 顺序**

  使用多 combo、非连续和乱序 trial index；验证每个原始 trial 的输出落回正确位置，
  完成位图全部为 1。

- [ ] **Step 4：覆盖数值边界**

  CPU/GPU 同时验证精确 tie、近 tie、全负相关、零 norm、NaN、Inf 和退化 synamp。
  Inf/preflight 失败不得改变旧 intermediates；NaN mask 与退化行为必须一致。

**Verify:**

```bash
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    FM_FORWARD_BACKEND=cpu julia --project=. tests/stages/forward_test.jl
FM_FORWARD_EXE="$PWD/forward/build-cuda/forward" \
    FM_FORWARD_BACKEND=cuda julia --project=. tests/stages/forward_test.jl
```

**建议 commit：** `test: 覆盖 CUDA parity 与分批边界`

______________________________________________________________________

## Task 9：完成 fallback、错误传播与事务集成测试

**Files:**

- Modify: `tests/stages/forward_test.jl`

- Modify: `forward/tests/hdf5_commit_test.cpp`

- Modify: `forward/src/main.cpp`

- [ ] **Step 1：覆盖 backend fallback**

  在无可见 device 环境验证 CUDA build 的 auto 回退 CPU、显式 cuda 失败；用可控 runtime
  测试替身验证其他初始化错误不回退。CPU-only 显式 cuda 始终失败。

- [ ] **Step 2：覆盖执行期 CUDA 错误**

  用内部测试接口在首个 batch 和中间 batch 注入 launcher/runtime 错误；必须非零退出，
  不得继续 CPU，也不得提交部分 host 输出。

- [ ] **Step 3：组合 preflight/CUDA/HDF5 失败与恢复**

  每个测试先写入带 marker 的旧 `/intermediates`，触发失败后重新打开文件。断言旧结果
  可恢复、没有把未计算零值写入 final、下次正常运行可成功替换。

- [ ] **Step 4：重复运行幂等性**

  CPU 与 CUDA 各连续运行两次，结果完全一致且没有 HDF5 diagnostic、tmp 或 backup
  残留。

**Verify:**

```bash
ctest --test-dir forward/build-cpu --output-on-failure
ctest --test-dir forward/build-cuda --output-on-failure
```

**建议 commit：** `test: 覆盖 CUDA fallback 与 HDF5 恢复`

______________________________________________________________________

## Task 10：运行 synthetic CUDA 全流程与 sanitizer

**Files:**

- Modify: `tests/stages/e2e_test.jl`

- Modify: `tests/stages/test_util.jl`

- [ ] **Step 1：增加 GPU opt-in e2e**

  将 driver 使用的默认 `forward/build/forward` clean configure 为 CUDA-enabled，运行
  真实 input→preprocess→forward→assess→output→report。日志必须包含实际 backend
  `cuda` 和 GPU 名称，不能只依据请求值判断。

- [ ] **Step 2：验证最终科学结果**

  CPU/CUDA 都恢复 synthetic 真值 MT、10 km depth、0.2 s duration；最终 misfit 与 CPU
  基线一致。自动 batch 和强制多 batch 至少各跑一次。

- [ ] **Step 3：运行 compute-sanitizer**

  对小型、强制多 batch fixture 运行 memcheck；不得报告越界、未初始化访问或泄漏。
  sanitizer 用小 fixture，不重复耗时的 455,544-trial e2e。

- [ ] **Step 4：记录性能，不设门槛**

  保存 CPU evaluation、CUDA context/H2D/kernel/D2H 和 forward 总耗时，以及实际加速比。
  不因 GPU 未快于 CPU 判定失败。

**Verify:**

```bash
cmake -S forward -B forward/build \
    -DFM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build forward/build -j8
FM_FORWARD_BACKEND=cuda julia --project=. tests/stages/e2e_test.jl
compute-sanitizer --tool memcheck "$PWD/forward/build-cuda/forward" \
    --backend cuda --cuda-batch-trials 7 <database.h5> <status_N.h5>
```

**建议 commit：** `test: 验证 CUDA synthetic 全流程`

______________________________________________________________________

## Task 11：同步文档并完成最终验收

**Files:**

- Modify: `forward/AGENTS.md`

- Modify: `doc/design.md`

- Modify: `doc/modules/misfit_kernel.md`

- Modify: `doc/modules/mt_utils.md`

- Modify: `doc/stages/forward.md`

- Modify: `doc/roadmap.md`

- [ ] **Step 1：更新架构和构建文档**

  记录 CPU-only/CUDA clean build、architecture 参数、CLI/环境变量优先级、fallback 状态
  表、显存生命周期、batch 公式、事务恢复和日志字段。

- [ ] **Step 2：修正已知过期说明**

  `misfit_kernel.md` 改为当前 XCorr P+S；`mt_utils.md` 说明 C++ `sdr_to_mt` 正在使用；
  `design.md` 与 `forward/AGENTS.md` 同时描述 OpenMP/CUDA 后端。

- [ ] **Step 3：执行 clean build 矩阵**

  删除任务专用的临时 build 目录后，从空目录分别 configure/build CPU-only 与 CUDA。
  CPU-only 环境验证不需要 NVCC；CUDA 构建显式使用 architecture 120。

- [ ] **Step 4：执行完整回归**

  运行 CTest、forward CPU/GPU tests、assess/output stage tests、CPU/GPU e2e、format check
  和 sanitizer。记录未运行项及原因，不以 skip 代替声明有 GPU 环境中的失败。

- [ ] **Step 5：审查 diff**

  确认没有新 operator、schema 改动、异步 stream、fast-math、调试代码、测试数据或
  未使用抽象。若用户要求提交，再按逻辑组提交。

**Verify:**

```bash
bash format.sh
bash format.sh --check
ctest --test-dir forward/build-cpu --output-on-failure
ctest --test-dir forward/build-cuda --output-on-failure
FM_FORWARD_EXE="$PWD/forward/build-cpu/forward" \
    julia --project=. tests/stages/forward_test.jl
FM_FORWARD_EXE="$PWD/forward/build-cuda/forward" \
    FM_FORWARD_BACKEND=cuda julia --project=. tests/stages/forward_test.jl
julia --project=. tests/stages/assess_test.jl
julia --project=. tests/stages/output_test.jl
```

**建议 commit：** `docs: 完善 forward CUDA 使用与验证说明`

______________________________________________________________________

## 最终完成条件

- [ ] CPU-only clean build 不探测/链接 CUDA，现有调用与结果保持兼容。
- [ ] CUDA clean build 在 architecture 120 上成功，同一 binary 可强制 CPU/CUDA。
- [ ] CLI、环境变量优先级和 fallback 状态矩阵全部通过。
- [ ] 所有输入在分配/计算前完成 preflight，所有 trial 恰好计算一次。
- [ ] CPU/GPU `|Δcc_max| ≤ 1e-9`，`best_lag`、NaN mask、退化行为完全一致。
- [ ] 自动 batch、1、质数、残 batch、多 combo 和乱序 trial 全部通过独立 reference。
- [ ] preflight、CUDA 或 HDF5 任意失败后，旧 `/intermediates` 可恢复。
- [ ] synthetic CUDA e2e 恢复相同 MT、depth、duration，日志证明实际使用 CUDA。
- [ ] compute-sanitizer 无错误；计时结果已记录但不设性能门槛。
- [ ] 文档与实现一致，`bash format.sh --check` 通过，无临时文件残留。

## 建议提交分组

1. `test: 参数化 forward 阶段测试入口`
1. `refactor: 提取共享 XCorr work-item`
1. `fix: 严格校验并原子化 forward 输出`
1. `feat: 增加 forward CUDA 构建与分批执行`
1. `test: 覆盖 CUDA parity、恢复与全流程`
1. `docs: 完善 forward CUDA 使用与验证说明`
