# Stage: `driver.sh` — Pipeline orchestration

## Role

按固定阶段顺序运行管道，管理 status 目录、迭代决策和日志。

```text
input → [preprocess → forward → assess] → output → report
```

## Usage

```bash
bash driver.sh --data-dir <dir> [--trial-budget <N>]
```

`<dir>/config.jl` 必须存在。`driver.sh` 导出 `DATA_DIR`，创建
`status/`，并将 `input.jl` 生成的 `status_0.h5` 移入该目录。
`--trial-budget <N>` 设置每轮 Cartesian trial 数上限；省略时使用完整搜索空间。

## Loop contract

- `preprocess.jl` 在最新 status 中生成 trials。
- forward 写 `/intermediates`，`assess.jl` 写 `/misfits` 和 `.decision.txt`。
- `.decision.txt` 为空表示收敛；非空表示继续下一轮。
- 当前 assess 第一轮即收敛；多迭代策略待重新设计。

## Backend selection

`FM_FORWARD_EXE` 可指定 CPU-only 或 CUDA build。`FM_FORWARD_BACKEND`、
`FM_CUDA_BATCH_TRIALS` 由 forward 解析，详见 `doc/stages/forward.md`。

## Outputs

- `database.h5`、`status/status_0.h5`、`output.h5`
- `result.toml`、`report.md`、`.decision.txt`
- `driver.log` 及各 Julia stage 日志
