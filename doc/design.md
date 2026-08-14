# Design: 震源机制反演管道

## Overview

Julia 数据接入 + 预处理（Layer 0 共享预处理 + 算子 reductions），HDF5 数据交换。已完成：数据接入 (`input.jl`)、Misfit 算子（Xcorr 活跃；Polarity/Psr 已实现但 **deferred**，未注册实例）、aggregate 两级聚合、试次生成、assess 聚合 + 收敛决策、output 编译、driver.sh 全管道贯通（单迭代闭环，**XCorr-only**）。待开发：assess 权重聚合/网格细化（多迭代闭环）。

## Project Layout

```
scripts/        Flat stage scripts (input/preprocess/assess/output — 全部实现)
shared/         Julia packages by function (not stage)
  io/           (IO)       HDF5 I/O abstractions
  mt/           (MT)       SDR ↔ MT conversion
  grid/         (Grid)     Trial generation + grid refinement
  signal/       (Signal)   Waveform preprocessing (filtering, trimming)
  config/       (Config)   Pipeline configuration interface
  misfit/       (Misfit)   Misfit operator package（Xcorr 活跃；Polarity/Psr 模板 + 输出字段常量，deferred）
  aggregate/    (Aggregate) Output extractor + composer 注册表
  stage_log/    (StageLog)  Per-stage logging
forward/        C++ forward stage (OpenMP CPU) — kernel 产出中间产物
config_sample.jl   Template pipeline configuration
```

## Pipeline

```
input.jl (once) → loop: [preprocess → forward → assess → [repeat]] → output
```

| Stage | Role | Status |
|--------------------------|---------------------------------------------------------------------------|-------------------------------|
| `input.jl` | 数据接入 → `database.h5`；初始 strategy → `status_0.h5` | 已完成 |
| `preprocess.jl` | 从 strategy 生成 trials（全参数 paraspace 索引化）→ `status_{N}.h5:/trials` | 已完成 |
| forward (C++/OpenMP CPU) | kernel 重计算，产出**中间产物** → `status_{N}.h5:/intermediates/` | 已实现 |
| `assess.jl` | extractor + composer → `/misfits/`；收敛决策 | 已完成（权重聚合/网格细化待做） |
| `output.jl` | 编译最终结果 → `output.h5` | 已完成（加权聚合后完善） |
| 编排层 (`driver.sh`) | 状态检测、阶段调用、循环控制 | 已实现（单迭代闭环） |

## Data Files

| File | Lifetime | Contents |
|-----------------|---------------|------------------------------------------------------------------------------------------------------------------|
| `database.h5` | Static | 预处理波形、格林函数各深度变体、模块预处理结果、**`/paraspace`**（展开的参数空间浮点数组）、`/config`（算法参数，无索引） |
| `status_{N}.h5` | Per-iteration | `/strategy`（整数索引指向 `/paraspace`）、trials、`/intermediates`（kernel 中间产物）、`/misfits`（最终 misfit 值） |
| `output.h5` | Final | 最佳拟合参数、不确定性、逐阶段/台站分解 |
| `config.jl` | Bootstrap | 失配模块列表、频带、深度、初始网格参数、数据接口实现 |

## HDF5 三层分离：值 / 索引 / 参数

| 文件 | 组 | 职责 | 示例 |
|-----------------|--------------|---------------------------------------------------------|--------------------------------------------------------------------------|
| `database.h5` | `/paraspace` | **存值**：展开的浮点数组 | `strike[72]`, `dip[19]`, `rake[37]`, `depth[3]`, `frequency[2]` |
| `database.h5` | `/config` | **参数**：模块设置、算法元数据，**不含索引或浮点参数值** | `misfit_modules`, `{ModuleName}/trim`, `max_lag_periods` |
| `status_{N}.h5` | `/strategy` | **网格定义**：SDR 展开轴参数 + 整数索引指向 `/paraspace` | `strike0/dstrike/nstrike…`, `depth_indices`, `freq_indices`, `iteration` |

规则：`/paraspace` 只存浮点值；`/config` 只存参数；`/strategy` 只存整数索引。三者永不混杂。`freq_indices` 选搜索频带（1..N_bands），模块经 `/config/{ModuleName}/{band_low,band_high}` 指向 `/paraspace/frequency`；`depth_indices` 选搜索深度。

## Key Design Rules

1. **所有频带变体在 input.jl 中预计算** — 写入 `database.h5`，后续阶段无运行时滤波。
1. **格林函数外部预计算** — 由 `input.jl` 加载，管道内不计算。
1. **配置通过 `config.jl` 引导** — 仅 `input.jl` 读取，全部写入 `database.h5`，后续阶段从 HDF5 读取。
1. **HDF5 schema 是阶段间接口契约** — schema 变更需协调的阶段更新。
1. **Flat scripts** — 阶段脚本顶层直列执行，无 `main()` 包装；私有辅助函数扁平化深层嵌套（自包含、从属主流程）。
1. **`/strategy` 仅含网格定义** — 无迭代状态字段（weights, best-fit, convergence）；状态由各阶段自行管理。
1. **forward 无状态** — 读数据 + trials，写中间产物到 `/intermediates/`；无权重/聚合/策略/输出变换。
1. **三层分离** — `/paraspace` 存值，`/config` 存参数，`/strategy` 存索引。
1. **Misfit 三层分解** — Misfit = Operator × Phase × Output；C++ kernel 产出中间产物，Julia extractor/composer 产出最终 misfit。详见 `doc/misfit-decomposition.md`。

## Misfit 分解（摘要）

Misfit = **Operator × Phase × Output**，完整设计见 `doc/misfit-decomposition.md`。

- **Level 1（Base）**：Operator（XCorr/Polarity）× Phase（P/S）× Output（cc_max/best_lag/...）。kernel 消费预处理数据，产出中间产物到 `/intermediates/{Operator}{Phase}[_{channel}]/`。同一 (Operator, Phase) 可派生多个 Base misfit，共享一次 kernel 运行。
- **Level 2（Composed）**：Aggregate Operator（StdDev/...）× Base misfit 集合 × Output。纯 Julia，消费 Level 1 值聚合，不触及波形/GF。

**C++/Julia 边界**：C++ 只重计算 → `/intermediates/`；Julia 做语义解释 → `/misfits/`。

`shared/misfit/` 为正式 Julia package，每个算子是 `module`（输出字段常量 + 模板桩 + 预处理）；`using Misfit` 后 `Misfit.Xcorr.CC_MAX` 形式指定输出，注册时校验 `output ∈ operator.outputs()`。

Config 接口（XCorrS-only 现状）：

```julia
Config.use_misfit!(:XcorrS, operator = Misfit.Xcorr, phase = "S", output = Misfit.Xcorr.CC_MAX)
```

多实例/多输出扩展见 `doc/misfit-decomposition.md`（多模块注册已于 2026-08-09 XCorrS-only 清理时移除）。

## Dimension Symbols

| Symbol | Description | Typical Value |
|--------------|--------------------------------------------------------|---------------|
| `N_stations` | Unique physical stations | 10–30 |
| `N_channels` | Unique (station, channel) pairs | 30–90 |
| `N_phases` | Phase entries (channel + wave type: P/S) | 20–180 |
| `N_depths` | Depth levels for Greens | 10–40 |
| `N_bands` | Frequency band combinations | configurable |
| `N_modules` | Active misfit module instances (Operator×Phase×Output) | 2–6 |
| `N_trials` | Trials per iteration | 10–100000 |
