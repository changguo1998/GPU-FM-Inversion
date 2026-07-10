# Roadmap — 震源机制反演管道（从头开发）

当前为从头开发第一阶段，仅完成数据接入与初始化（`input.jl`）。以下为后续阶段规划。

## Legend

- [x] Completed
- [ ] Not started

______________________________________________________________________

## 已完成

| Task | 说明 |
|-----------------------------------------------------------------------------|---------------------|
| [x] IO module — HDF5 read/write, type structs, geophysics utilities | `shared/io/` |
| [x] MT module — SDR↔MT conversion | `shared/mt/` |
| [x] Grid module — trial generation + grid refinement | `shared/grid/` |
| [x] Signal module — waveform preprocessing | `shared/signal/` |
| [x] Aggregate module — misfit aggregation | `shared/aggregate/` |
| [x] Config module — interface declarations | `shared/config/` |
| [x] StageLog module — per-stage logging | `shared/stage_log/` |
| [x] `input.jl` — data ingestion, preprocessing, database + initial strategy | `scripts/input.jl` |

______________________________________________________________________

## Phase 1: 后续阶段开发

| Task | Priority | 说明 |
|-----------------------------------------------|----------|-----------------------------------|
| [ ] `preprocess.jl` — 从 strategy 生成 trials | P0 | 写 status_N.h5 /trials group |
| [ ] forward 阶段 — 失配计算 | P0 | 需设计新接口（是否继续用 C++ 待定） |
| [ ] `assess.jl` — 加权/聚合/网格细化 | P0 | 读 misfits，写 refined strategy |
| [ ] `output.jl` — 输出编译 | P0 | 读所有 status 文件，写 output.h5 |
| [ ] `driver.sh` — 管道编排 | P0 | 5 阶段循环 + exit code 检测 |

## Phase 2: 验证

| Task | 说明 |
|-----------------------|---------------------------------------|
| [ ] input.jl 单元测试 | 验证 database.h5 + status_0.h5 schema |
| [ ] 端到端集成测试 | 全管道贯通测试 |

## Phase 3: 高级模块

| Task | 说明 |
|----------------------------|------|
| [ ] PSR module | 延期 |
| [ ] AbsShift / RelShift | 延期 |
| [ ] 非交互 operator prompt | 延期 |
