# Roadmap - 震源机制反演管道（从头开发）

当前为从头开发第一阶段，仅完成数据接入与初始化（`input.jl`）。以下为后续阶段规划。

## Legend

- [x] Completed
- [ ] Not started

______________________________________________________________________

## 已完成

| Task | 说明 |
|-----------------------------------------------------------------------------|---------------------|
| [x] IO module - HDF5 read/write, type structs, geophysics utilities | `shared/io/` |
| [x] MT module - SDR↔MT conversion | `shared/mt/` |
| [x] Grid module - trial generation + grid refinement | `shared/grid/` |
| [x] StageLog module - per-stage logging | `shared/stage_log/` |
| [x] `input.jl` - data ingestion, preprocessing, database + initial strategy | `scripts/input.jl` |

______________________________________________________________________

## 设计基线

| 设计文档 | 内容 |
|-------------------------------|----------------------------------------------------------------------------------------|
| `doc/design.md` | 管道总体设计（值/索引/参数三层分离、阶段划分） |
| `doc/schema.md` | HDF5 schema 完整规范 |
| `doc/misfit-decomposition.md` | Misfit 三层分解（Operator × Phase × Output），C++/GPU 中间产物 + Julia extractor/composer |

______________________________________________________________________

## Phase 1: Misfit 分解框架

落地 Operator × Phase × Output 三层分解。详见 `doc/misfit-decomposition.md` §12。

| Task | Priority | 说明 |
|--------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ ] Misfit package 化 + 输出字段常量 | P0 | `shared/misfit/` 转 package；XCorr/Polarity 加 `outputs()` + 常量；`Config.use_misfit!` 新签名（operator/output）+ 校验；`config_sample.jl` 适配 |
| [ ] forward kernel 产出中间产物 | P0 | `xcorr_kernel.h` 加 `best_lag` 输出；`polarity_kernel.h` 输出 `syn_sign`+`dot_value`；`main.cpp` 写 `/intermediates/` 而非 `/misfits/`，按 (operator,phase,channel) 去重跑 kernel |
| [ ] aggregate extractor + composer | P0 | `shared/aggregate/` 新 package；EXTRACTORS/COMPOSERS 注册表；`scripts/assess.jl` 实现 extract + compose |
| [ ] RelShift 组合示例 | P1 | StdDev composer 实现；`config_sample.jl` 加 AbsShiftP/SH/SV + RelShift 示例；`doc/schema.md` 更新 `/intermediates` |

## Phase 2: 管道贯通

| Task | Priority | 说明 |
|-----------------------------------------------|----------|-----------------------------------------------------------------------------|
| [ ] `preprocess.jl` - 从 strategy 生成 trials | P0 | 写 status_N.h5 /trials group |
| [ ] `assess.jl` - 加权/聚合/网格细化 | P0 | 读 misfits，写 refined strategy（在 Phase 1 extract/compose 基础上加权重聚合） |
| [ ] `output.jl` - 输出编译 | P0 | 读所有 status 文件，写 output.h5 |
| [ ] `driver.sh` - 管道编排 | P0 | 5 阶段循环 + exit code 检测 |

## Phase 3: 验证

| Task | 说明 |
|---------------------------------|------------------------------------------------|
| [ ] input.jl 单元测试 | 验证 database.h5 + status_0.h5 schema |
| [ ] forward 中间产物测试 | 验证 /intermediates/ 字段（cc_max, best_lag 等） |
| [ ] assess extract/compose 测试 | 验证 extractor 变换 + composer 聚合 + 拓扑排序 |
| [ ] 端到端集成测试 | 全管道贯通测试（含 RelShift 组合） |

## Phase 4: 高级模块

| Task | 说明 |
|----------------------------|--------------------------------|
| [ ] PSR module | 延期（kernel 已存在，待接入框架） |
| [ ] 非交互 operator prompt | 延期 |
