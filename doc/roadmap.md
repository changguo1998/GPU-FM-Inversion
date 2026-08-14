# Roadmap - 震源机制反演管道（从头开发）

已完成数据接入与初始化（`input.jl`）及 Misfit 分解框架（Phase 1，2026-07-16 merge）与算子预处理重构（2026-07-18 merge）。以下为后续阶段规划。

## 开发路线

围绕一个事例数据（`examples/synthetic`），先以 **XCorr 为唯一目标函数** 打通并打磨全流程
（input → preprocess → forward → assess → output），验证收敛后再扩展其他算子（Polarity/Psr —— 见 Phase 4 / 最后阶段）。

## Legend

- [x] Completed
- [ ] Not started

______________________________________________________________________

## 已完成

| Task | 说明 |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [x] IO module - HDF5 read/write, type structs, geophysics utilities | `shared/io/` |
| [x] MT module - SDR↔MT conversion | `shared/mt/` |
| [x] Grid module - trial generation + grid refinement | `shared/grid/` |
| [x] StageLog module - per-stage logging | `shared/stage_log/` |
| [x] `input.jl` - data ingestion, preprocessing, database + initial strategy | `scripts/input.jl` |
| [x] Misfit 算子 package - Xcorr/Polarity/Psr (process + outputs) | `shared/misfit/`（Polarity/Psr **deferred** — XCorr-only 模式，2026-08-09） |
| [x] Aggregate 两级聚合 - extractors + composers + StdDev | `shared/aggregate/` + `scripts/assess.jl` |
| [x] Layer 0 共享预处理 - demean/detrend/taper/bandpass per band + per-lag/PSR reductions | 07-18 feat/operator-preprocessing (c030626) |

______________________________________________________________________

## 设计基线

| 设计文档 | 内容 |
|-------------------------------|-------------------------------------------------------------------------------------------|
| `doc/design.md` | 管道总体设计（值/索引/参数三层分离、阶段划分） |
| `doc/schema.md` | HDF5 schema 完整规范 |
| `doc/misfit-decomposition.md` | Misfit 三层分解（Operator × Phase × Output），C++/OpenMP 中间产物 + Julia extractor/composer |

______________________________________________________________________

## Phase 1: Misfit 分解框架 ✅ (2026-07-16 完成)

落地 Operator × Phase × Output 三层分解。详见 `doc/misfit-decomposition.md` §12。

| Task | Priority | 说明 |
|--------------------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [x] Misfit package 化 + 输出字段常量 | P0 | `shared/misfit/` 转 package；XCorr/Polarity 加 `outputs()` + 常量；`Config.use_misfit!` 新签名（operator/output）+ 校验；`config_sample.jl` 适配 |
| [x] forward kernel 产出中间产物 | P0 | `xcorr_kernel.h` 加 `best_lag` 输出；`polarity_kernel.h` 输出 `syn_sign`+`dot_value`；`main.cpp` 写 `/intermediates/` 而非 `/misfits/`，按 (operator,phase,channel) 去重跑 kernel |
| [x] aggregate extractor + composer | P0 | `shared/aggregate/` 新 package；EXTRACTORS/COMPOSERS 注册表；`scripts/assess.jl` 实现 extract + compose |
| [x] RelShift 组合示例 | P1 | StdDev composer 实现；`config_sample.jl` 加 AbsShiftP/S + RelShift 示例；`doc/schema.md` 更新 `/intermediates`（**示例已于 2026-08-09 XCorrS-only 清理时移除**；StdDev composer 保留于 `shared/aggregate/`） |

## Phase 2: 管道贯通

| Task | Priority | 说明 |
|-----------------------------------------------|----------|---------------------------------------------------------------------------------------------------------------------------|
| [x] `preprocess.jl` - 从 strategy 生成 trials | P0 | 写 status_N.h5 /trials group（已实现，2026-08-07） |
| [ ] `assess.jl` - 加权/聚合/网格细化 | P0 | 读 misfits，写 refined strategy（extract+compose + 收敛决策已实现；**权重聚合/网格细化待补**） |
| [x] `output.jl` - 输出编译 | P0 | 读所有 status 文件，写 output.h5（已实现最小版，2026-08-07：Xcorr 主 misfit 选 best；加权聚合落地后完善） |
| [x] `driver.sh` - 管道编排 | P0 | input →（preprocess→forward→assess 循环）→ output 全链已打通（2026-08-07，单迭代收敛，exit code 检测）；多迭代细化待 assess 落地 |

## Phase 3: 验证

状态：package 单元测试已通过（io 97、mt 95、grid 34、config、signal、misfit、aggregate；从根环境 `julia --project=. -e 'include("shared/<pkg>/test/runtests.jl")'` include 运行）。下列为待补项。

| Task | 说明 |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [x] input.jl 单元测试 | 验证 database.h5 + status_0.h5 schema（`tests/stages/input_test.jl`，107 断言，2026-08-14） |
| [x] preprocess.jl 单元测试 | 验证 /trials 索引化笛卡尔积生成（`tests/stages/preprocess_test.jl`，20 断言，2026-08-14） |
| [x] forward 中间产物测试 | 验证 /intermediates/ 字段（`tests/stages/forward_test.jl`：独立 Julia 参考严格对比 cc_max/best_lag ≤1e-9 + 幂等 + 无 DIAG；含 C++ kernel 修复：maxlag 配置推导、per-lag synamp、station_idx 0-based、synthetic_data.jl MT 公式 bug（sin(2d)→sind(2d)）修复，445 断言，2026-08-14） |
| [x] assess extract/compose 测试 | 验证 extractor 变换 + composer 聚合 + 拓扑排序（`tests/stages/assess_test.jl`：extract misfit = 1−cc_max + 收敛决策 + 不动 strategy/trials，14 断言；compose 拓扑由 `shared/aggregate` 包测试覆盖，2026-08-14） |
| [x] output.jl 单元测试 | 验证 output.h5 五组 schema + best 选择（`tests/stages/output_test.jl`：solution == argmin mean-misfit / cross_correlation == cc_max@best / summary counts，21 断言，2026-08-14） |
| [ ] 端到端集成测试 | 全管道贯通测试（XCorrS-only 单模块） |

## Phase 4: 高级模块

| Task | 说明 |
|----------------------------|-----------------------------------------------------------------------------|
| [x] PSR module | 算子已实现（`Psr.jl` + tests，07-18）；**deferred**（XCorr-only 模式，未注册实例） |
| [ ] 非交互 operator prompt | 延期 |
