# Roadmap - 震源机制反演管道

项目当前围绕 `examples/synthetic` 的多目标单轮全流程开发。

## 已完成

| 范围 | 实现 |
|--------------|---------------------------------------------------------------------------------------------|
| 数据与预处理 | `input.jl`、Layer 0 共享预处理、XCorr P/S reductions、Gaussian STF duration 候选 |
| 目标函数 DSL | 六个数学原语、表达式序列化、基础 XCorr 编译、通用 Expression 编译与 assess 求值 |
| 搜索空间 | strike/dip/rake/depth/frequency/duration 全参数索引化、trials 笛卡尔积与预算约束初始采样规划 |
| Forward | OpenMP CPU/CUDA 共用 XCorr/能量/振幅/符号公式；CUDA 显存复用、分批、preflight 与 HDF5 事务提交 |
| Assess | DSL 通用表达式求值、XCorr、signed lag、PSR、归一化极性、composer 框架与单轮收敛决策 |
| 输出 | `output.h5`、机器可读 `result.toml`、人工可读 `report.md` |
| 编排 | `driver.sh` 贯通 input → preprocess → forward → assess → output → report |
| 验证 | package/stage 测试、CPU/GPU parity、CUDA e2e、compute-sanitizer 与失败恢复 |

## 当前验收基线

- 455,544 trials，候选 STF σ = `[0.1, 0.2, 0.3] s`。
- P+S best = `(210, 30, 90) @ 10 km, σ=0.2 s`，misfit ≈ `6.993e-5`。
- 该解与真值 `(30, 60, 90)` 为同一 moment tensor 的两个节面表示；P/S best lag 均为 0。
- CPU/GPU `cc_max` 差异 ≤ `1e-9`，`best_lag` 完全一致。

## 待开发

1. **Assess 配置化权重**：当前仅支持等权归一化聚合。
1. **多迭代处理**：接入预算规划，设计基于 misfit 的区域保留、裁剪与下一轮调度。

## 长期文档

| 文档 | 内容 |
|--------------------------------|---------------------------------------|
| `doc/design.md` | 管道总体设计与阶段边界 |
| `doc/schema.md` | HDF5 schema 契约 |
| `doc/misfit-decomposition.md` | Operator × Phase × Output 分解 |
| `doc/stages/forward.md` | CPU/CUDA forward 后端、分批、错误与性能 |
| `doc/modules/misfit_kernel.md` | XCorr kernel 公式与测试契约 |
