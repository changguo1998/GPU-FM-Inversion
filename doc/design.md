# Design: 震源机制反演管道（从头开发）

## Overview

震源机制反演管道。当前为从头开发第一阶段，仅完成数据接入与初始化。Julia 数据接入 + 预处理，HDF5 数据交换。后续阶段（试次生成、失配计算、加权聚合、输出编译）待开发。

## Current Project Layout

```
scripts/        Flat stage scripts (当前仅 input.jl)
shared/         Julia packages by function (not stage)
  io/           (module: IO)      ← HDF5 I/O abstractions
  mt/           (module: MT)      ← SDR ↔ MT conversion
  grid/         (module: Grid)    ← Trial generation + grid refinement
  signal/       (module: Signal)  ← Waveform preprocessing (filtering, trimming)
  config/       (module: Config)  ← Pipeline configuration interface
  stage_log/    (module: StageLog) ← Per-stage logging
config_sample.jl   Template pipeline configuration
```

## Current Stage

```
scripts/input.jl  (once) → database.h5 + status_0.h5
```

`input.jl` 已完成：读取 `config.jl` 配置，通过 `Config.load_*()` 接口加载外部数据（波形、台站、震相、格林函数），预处理后写入 `database.h5`，并生成初始搜索策略 `status_0.h5`。

## Future Pipeline (规划)

```
input (once) → loop: [preprocess → forward → assess → [repeat]] → output
```

| Stage | Role |
|-----------------|---------------------------------------------------------------|
| `input.jl` | 已完成。数据接入 → `database.h5`；初始 strategy → `status_0.h5` |
| `preprocess.jl` | 待开发。从 strategy 生成 trials → `status_{N}.h5` |
| forward | 待设计。失配计算（是否用 C++/GPU 待定） |
| `assess.jl` | 待开发。加权、聚合、网格细化、operator prompt |
| `output.jl` | 待开发。编译最终结果 → `output.h5` |
| 编排层 | 待设计。状态检测、阶段调用、循环控制 |

## Data Files

| File | Lifetime | Contents |
|-----------------|---------------|----------------------------------------------------------------------------------------------------------------------|
| `database.h5` | Static | 预处理波形、格林函数各深度变体、模块预处理结果、**`/paraspace`**（展开的参数空间浮点数组）、`/config`（算法参数，**无索引**） |
| `status_{N}.h5` | Per-iteration | **`/strategy`**（整数索引指向 `/paraspace`），后续阶段追加 trials、misfits |
| `output.h5` | Final | 最佳拟合参数、不确定性、逐阶段/台站分解（待实现） |
| `config.jl` | Bootstrap | 用户提供：失配模块列表、频带、深度范围、初始网格参数、数据接口实现 |

## HDF5 层级设计

### 三层分离：值 / 索引 / 参数

| 文件 | 组 | 职责 | 示例 |
|-----------------|--------------|-----------------------------------------------------------|------------------------------------------------------------------------|
| `database.h5` | `/paraspace` | **存值**：展开的浮点数组，所有参数空间维度 | `strike[71]`, `dip[19]`, `rake[37]`, `depth[3]`, `frequency[2]` |
| `database.h5` | `/config` | **参数**：算法元数据、模块设置，**不含任何索引或浮点参数值** | `misfit_modules`, `n_bands`, `xcorr/maxlag_factor` |
| `status_{N}.h5` | `/strategy` | **索引**：整数索引指向 `/paraspace`，定义当前迭代搜索范围 | `depth_indices[3]`, `freq_low_idx[1]`, `freq_high_idx[1]`, `iteration` |

规则：

- `/paraspace` 存实际浮点值（`Float64[N]`），永不存入整数索引
- `/config` 存模块参数和元数据，**永不存储整数索引或浮点参数值**
- `/strategy` 存整数索引（`Int32[N]`），永不存原始浮点值
- `freq_low_idx` / `freq_high_idx` 定义每个频带的低切/高切在 `frequency` 数组中的位置
- `depth_indices` 选择搜索哪些深度
- 没有 `freq_indices` — 所有频带由 `freq_low_idx` / `freq_high_idx` 隐式定义

## Key Design Rules (Current)

1. **所有频带变体在 input.jl 中预计算** — 写入 `database.h5`，后续阶段无运行时滤波。
1. **格林函数外部预计算** — 由 `input.jl` 加载，管道内不计算格林函数。
1. **配置通过 `config.jl` 引导** — 实现 `Config` 模块接口，仅 `input.jl` 读取。所有配置写入 `database.h5`；后续阶段从 HDF5 读取。
1. **HDF5 schema 是阶段间接口契约** — schema 变更需要协调的阶段更新。
1. **Flat scripts** — 阶段脚本无 `function` 定义，顶层直列执行。
1. **`/strategy` 仅含网格定义** — 无迭代状态字段（weights, best-fit, convergence）。状态由各阶段自行管理。
1. **`forward` 模块无状态** — 读数据 + trials，写原始 misfits。不涉及权重、聚合、策略。
1. **shared packages** — 工具代码在 `shared/` Julia 包中，通过 `using` 导入。
1. **三层分离** — `/paraspace` 存值，`/config` 存参数，`/strategy` 存索引。三者永不混杂。

## Dimension Symbols

| Symbol | Description | Typical Value |
|--------------|------------------------------------------|---------------------|
|| `N_stations` | Unique physical stations | 10–30 |
| `N_channels` | Unique (station, channel) pairs | 30–90 |
| `N_phases` | Phase entries (channel + wave type: P/S) | 20–180 |
| `N_depths` | Depth levels for Greens | 10–40 |
| `N_bands` | Frequency band combinations | configurable |
| `N_modules` | Active misfit modules | 2 (XCorr, Polarity) |
| `N_trials` | Trials per iteration | 10–100000 |
