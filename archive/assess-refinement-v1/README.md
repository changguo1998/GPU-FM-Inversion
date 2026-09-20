# Assess Refinement Strategy v1 (Archived)

归档日期：2026-09-20。

本目录保存已废弃的第一版 assess 网格细化方案，供历史追溯。该方案未接入
`scripts/assess.jl`，不得作为当前设计或可用接口引用。

## Archived contents

- `grid_refinement.jl`: `TrialResult`、`refine_strategy`、人工继续提示
- `grid_runtests.jl`: 归档时的 Grid 测试快照，包含旧细化测试
- `grid_AGENTS.md`: 归档时的 Grid 模块接口和规则说明
- `trial_gen.md`: 归档时的 trial generation/refinement 设计文档

## Old strategy summary

- SDR 步长减半，以最佳解为中心生成固定 `3×3×3` 网格
- depth/frequency 保留 misfit 不超过最佳值 1.2 倍的索引
- duration 原样保留
- 操作者通过 `y/N` 决定是否继续
- 下一轮拟写入 `status_{N+1}.h5`

该方案整体待重新设计，不应增量修改。
