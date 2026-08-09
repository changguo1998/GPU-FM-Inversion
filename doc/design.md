# Design: 震源机制反演管道（从头开发）

## Overview

震源机制反演管道。Julia 数据接入 + 预处理（Layer 0 共享预处理 + 算子 reductions），HDF5 数据交换。已完成：数据接入 (input.jl)、Misfit 算子、aggregate 两级聚合、assess.jl、preprocess.jl 试次生成、output.jl 输出编译、driver.sh 全管道贯通（单迭代闭环，**XCorr-only**）。Misfit 算子：Xcorr 活跃运行；Polarity/Psr 已实现但 **deferred**（未注册实例）。待开发：assess 权重聚合/网格细化（多迭代闭环）。

## Current Project Layout

```
scripts/        Flat stage scripts (input/preprocess/assess/output — 全部已实现)
shared/         Julia packages by function (not stage)
  io/           (module: IO)        ← HDF5 I/O abstractions
  mt/           (module: MT)        ← SDR ↔ MT conversion
  grid/         (module: Grid)      ← Trial generation + grid refinement
  signal/       (module: Signal)    ← Waveform preprocessing (filtering, trimming)
  config/       (module: Config)    ← Pipeline configuration interface
  misfit/       (module: Misfit)    ← Misfit 算子 package（Xcorr 活跃；Polarity/Psr 模板 + 输出字段常量，deferred）
  aggregate/    (module: Aggregate) ← Output extractor + composer 注册表
  stage_log/    (module: StageLog)  ← Per-stage logging
forward/        C++ forward stage (GPU) ← kernel 产出中间产物
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
|-------------------|---------------------------------------------------------------------------------------|
| `input.jl` | 已完成。数据接入 -> `database.h5`；初始 strategy -> `status_0.h5` |
| `preprocess.jl` | 待开发。从 strategy 生成 trials -> `status_{N}.h5` |
| forward (C++/GPU) | 已设计。kernel 产出**中间产物** -> `status_{N}.h5:/intermediates/`（见 Misfit 分解设计） |
| `assess.jl` | 待开发。Output extractor + Composer -> `status_{N}.h5:/misfits/`；加权、聚合、网格细化 |
| `output.jl` | 待开发。编译最终结果 -> `output.h5` |
| 编排层 | 待设计。状态检测、阶段调用、循环控制 |

## Data Files

| File | Lifetime | Contents |
|-----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------|
| `database.h5` | Static | 预处理波形、格林函数各深度变体、模块预处理结果、**`/paraspace`**（展开的参数空间浮点数组）、`/config`（算法参数，**无索引**） |
| `status_{N}.h5` | Per-iteration | **`/strategy`**（整数索引指向 `/paraspace`），后续阶段追加 trials、**`/intermediates`**（kernel 中间产物）、`/misfits`（最终 misfit 值） |
| `output.h5` | Final | 最佳拟合参数、不确定性、逐阶段/台站分解（待实现） |
| `config.jl` | Bootstrap | 用户提供：失配模块列表、频带、深度范围、初始网格参数、数据接口实现 |

## HDF5 层级设计

### 三层分离：值 / 索引 / 参数

| 文件 | 组 | 职责 | 示例 |
|-----------------|--------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `database.h5` | `/paraspace` | **存值**：展开的浮点数组，所有参数空间维度 | `strike[71]`, `dip[19]`, `rake[37]`, `depth[3]`, `frequency[2]` |
| `database.h5` | `/config` | **参数**：算法元数据、模块设置，**不含任何索引或浮点参数值** | `misfit_modules`, `{ModuleName}/trim`, `max_lag_periods` |
| `status_{N}.h5` | `/strategy` | **网格定义**：SDR 展开轴参数 + 整数索引指向 `/paraspace`，定义当前迭代搜索范围 | `strike0/dstrike/nstrike…`, `depth_indices[3]`, `freq_indices[2]`, `iteration` |

规则：

- `/paraspace` 存实际浮点值（`Float64[N]`），永不存入整数索引
- `/config` 存模块参数和元数据，**永不存储整数索引或浮点参数值**
- `/strategy` 存整数索引（`Int32[N]`），永不存原始浮点值
- `freq_indices` 选择搜索哪些频带（1..N_bands）；模块经 `/config/{ModuleName}/band_low`、`band_high` 指向 `/paraspace/frequency`
- `depth_indices` 选择搜索哪些深度
- 频率维度由 `/paraspace/frequency` 存离散值、`/config` 存模块频带索引对、`/strategy` 存迭代搜索的频带索引

## Key Design Rules (Current)

1. **所有频带变体在 input.jl 中预计算** — 写入 `database.h5`，后续阶段无运行时滤波。
1. **格林函数外部预计算** — 由 `input.jl` 加载，管道内不计算格林函数。
1. **配置通过 `config.jl` 引导** — 实现 `Config` 模块接口，仅 `input.jl` 读取。所有配置写入 `database.h5`；后续阶段从 HDF5 读取。
1. **HDF5 schema 是阶段间接口契约** — schema 变更需要协调的阶段更新。
1. **Flat scripts** — 阶段脚本执行时无 `main()` 包装，顶层直列执行；允许私有辅助函数扁平化深层嵌套（保持自包含、从属于主流程）。
1. **`/strategy` 仅含网格定义** — 无迭代状态字段（weights, best-fit, convergence）。状态由各阶段自行管理。
1. **`forward` 模块无状态** - 读数据 + trials，写**中间产物**到 `/intermediates/`（不产出最终 misfit）。不涉及权重、聚合、策略、输出变换。
1. **shared packages** — 工具代码在 `shared/` Julia 包中，通过 `using` 导入。
1. **三层分离** — `/paraspace` 存值，`/config` 存参数，`/strategy` 存索引。三者永不混杂。
1. **Misfit 三层分解** - Misfit = Operator × Phase × Output。C++/GPU kernel 产出中间产物（`/intermediates/`），Julia extractor/composer 产出最终 misfit（`/misfits/`）。详见 `doc/misfit-decomposition.md`。

## Misfit 分解设计

Misfit = **Operator × Phase × Output**，三者正交组合。完整设计见 `doc/misfit-decomposition.md`。

### 两级 Misfit

- **Level 1（Base）**：Operator（XCorr/Polarity）× Phase（P/S）× Output（cc_max/best_lag/...）。C++/GPU kernel 消费预处理数据，产出**中间产物**写入 `status_{N}.h5:/intermediates/{Operator}{Phase}[_{channel}]/`。同一 (Operator, Phase) 可派生多个 Base misfit，共享一次 kernel 运行。
- **Level 2（Composed）**：Aggregate Operator（StdDev/...）× Base misfit 集合 × Output。纯 Julia，消费 Level 1 已算好的 misfit 值做聚合，不触及波形/GF。

### C++/Julia 边界

```
C++ forward (GPU)  -> /intermediates/   （kernel 重计算）
Julia assess       -> /misfits/         （extractor + composer，语义解释）
```

C++ 只做重计算，Julia 做语义解释。边界为 HDF5 文件交换。

### Operator package

`shared/misfit/` 为正式 Julia package，每个算子是 `module`（含输出字段常量 + 模板桩 + 预处理）。`using Misfit` 后即可用 `Misfit.Xcorr.CC_MAX` 形式指定输出，IDE 可补全，注册时校验 `output ∈ operator.outputs()`。

### Config 接口

```julia
Config.use_misfit!(:XcorrP_CC, operator=Misfit.Xcorr, phase="P", output=Misfit.Xcorr.CC_MAX)
Config.use_misfit!(:AbsShiftP, operator=Misfit.Xcorr, phase="P", output=Misfit.Xcorr.BEST_LAG)
Config.use_misfit!(:RelShift,  operator=Aggregate.StdDev,
                  bases=[:AbsShiftP, :AbsShiftS],  # 三分量 Z/N/E 可用 channel 过滤，见 misfit-decomposition §9
                  output=Aggregate.StdDev.RELATIVE_OFFSET)
```

## Dimension Symbols

| Symbol | Description | Typical Value |
|--------------|------------------------------------------|---------------------|
|| `N_stations` | Unique physical stations | 10–30 |
| `N_channels` | Unique (station, channel) pairs | 30–90 |
| `N_phases` | Phase entries (channel + wave type: P/S) | 20–180 |
| `N_depths` | Depth levels for Greens | 10–40 |
| `N_bands` | Frequency band combinations | configurable |
| `N_modules` | Active misfit module instances (Operator×Phase×Output) | 2–6 |
| `N_trials` | Trials per iteration | 10–100000 |
