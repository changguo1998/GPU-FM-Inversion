# Design: Misfit 三层分解（Operator × Phase × Output）

> **状态 (2026-08-09)**: 管道运行 **XCorr-only**。Polarity/Psr 算子已实现
> （`shared/misfit/` + C++ kernels）但 **deferred**——示例 config 不注册实例，
> `input.jl` 对注册的 Polarity/Psr 实例显式报错。本文 Polarity/Psr 内容为设计
> 记录；恢复时按 git HEAD 367dfd1 前的注册与预处理接线重新拉起。

## 1. 动机

旧模块采用扁平命名（`XcorrP`、`XcorrS`、`PolarityP`），算子、震相、输出三者耦合：
kernel 只输出单一 misfit 值，`best_lag`（时间偏移）被丢弃；组合关系（如
`RelShift = StdDev(三分量 AbsShift)`）无法表达；`use_misfit!` 收裸 Symbol 无校验。

设计目标：

1. **三层正交分解**：Misfit = Operator × Phase × Output，独立组合。
1. **kernel 产出中间产物**：一次计算保留所有字段（cc_max + best_lag），不丢信息。
1. **Julia 做语义解释**：extractor 从中间产物取值变换；composer 聚合基础 misfit。
1. **输出字段编译期安全**：`output = Xcorr.CC_MAX`，注册时校验 `output ∈ operator.outputs()`。

非目标：不改 `input.jl` 预处理数据流；不引入 ccall 共享库（边界仍为 HDF5）；PSR 组合/extractor 未实现（sample config 不注册 Psr）。

## 2. 概念模型

```
Misfit = Operator × Phase × Output

  Level 1（Base）:    Operator（C++ kernel）× Phase（P/S）× Output → extractor 变换为 misfit
  Level 2（Composed）: Aggregate Operator（StdDev/...）× Base misfit 集合 × Output（纯 Julia）
```

同一 (Operator, Phase) 派生多个 Base misfit，共享一次 kernel 运行，各取不同字段。
Composer 按 station 对齐各 base misfit，跨分量聚合（如 StdDev → relative_offset）。

## 3. 架构（B+C 混合）

```
C++ forward (kernel 重计算) ──HDF5──▶ /intermediates/  ──HDF5──▶ Julia assess (语义解释) ──▶ /misfits/
```

**C++ 只做重计算，Julia 做语义解释。** 边界为 HDF5 文件交换。中间产物数据量适中
（典型 100 phases × 100k trials × 2 字段 ≈ 160MB），I/O 可接受。

## 4. 数据流

```
database.h5 (obs, gf, synamp, obs_norm2, ...)
   ▼
C++ forward：按 (operator, phase, channel) 去重跑 kernel → /intermediates/
   XcorrP/S:  cc_max, best_lag      PolarityP: syn_sign, dot_value
   ▼
Julia assess：
   Output Extractor:  cc_max → 1.0-cc_max；best_lag → best_lag*dt
   Composer:          std(shift_Z, shift_N, shift_E) per station → /misfits/
```

## 5. /intermediates schema（status\_{N}.h5）

按 **(operator, phase, channel)** 去重存储——共享 kernel 运行的实例共用一份。
组名：`{Operator}{Phase}[_{channel}]`（如 `XcorrP`、`XcorrS_N`）。

| 组 | 字段 | 类型 | 形状 | 含义 |
|--------------|-------------|---------|---------------------------|---------------------------------------------|
| `Xcorr{P,S}` | `cc_max` | Float64 | `[N_phases × N_trials]` | 最大归一化 CC 值 |
| `Xcorr{P,S}` | `best_lag` | Int32 | `[N_phases × N_trials]` | 相对 maxlag 偏移（采样点；extractor ×dt 转秒） |
| `PolarityP` | `syn_sign` | Int8 | `[N_stations × N_trials]` | 合成极性符号 (-1/0/1) |
| `PolarityP` | `dot_value` | Float64 | `[N_stations × N_trials]` | 原始点积值（置信度） |

kernel 改动即：在更新 `max_abs_cc` 处同时记录 `best_k`，输出 `cc_max_out` 与
`best_lag_out = best_k - maxlag`（不再写 `1.0 - cc_max`）。性能开销可忽略。

## 6. /config 元数据（database.h5）

`/config/{ModuleName}/`：`operator`（"Xcorr"/"Polarity"/"StdDev"）、`phase`（P/S，
Level 1）、`output`（"cc_max"/"best_lag"/"relative_offset"...）、`channel`（Z/N/E/""，
Level 1 可选）、`bases`（[k]，Level 2）、`is_composed`（0/1）。

## 7. Extractor 与 Composer 注册表

```julia
# shared/aggregate/src/extractors.jl（按 (operator, output) 查找变换）
EXTRACTORS[(:Xcorr, :cc_max)]     = (intermediate, ctx) -> 1.0 .- intermediate["cc_max"]
EXTRACTORS[(:Xcorr, :best_lag)]   = (intermediate, ctx) -> intermediate["best_lag"] .* ctx.dt
EXTRACTORS[(:Polarity, :syn_sign)]= (intermediate, ctx) -> (intermediate["syn_sign"] .== ctx.obs_pol) ? 0.0 : 1.0
EXTRACTORS[(:Polarity, :dot_value)] = (intermediate, ctx) -> abs.(intermediate["dot_value"])
# ctx 携带 dt、obs_pol、station_idx 等上下文（读自 database.h5）

# shared/aggregate/src/composers.jl（组合算子按 operator 注册，返回 Dict{Symbol,Matrix}）
COMPOSERS[:std_dev] = function (base_misfits, base_station_idx, ctx)
    # 按 station 收集各分量值，std/mean → (:relative_offset, :mean)
end
```

组合算子的 `outputs()`（如 `StdDev.outputs() = [:relative_offset, :mean]`）声明其可用聚合字段。

### channel 子选择（Level 1 可选维度）

三分量实例按 `channel` 过滤 phase 条目（`use_misfit!(:AbsShiftS_N, ..., channel = "N")`）；
未指定时处理该 phase_type 全 channels。P 波通常单分量（Z）无需过滤。

## 8. assess.jl 流程

```
1. 读 status_N.h5:/intermediates/  2. 读 database.h5:/config/{Module}/（operator/phase/output/bases/is_composed）
3. 读 database.h5:/station, /{Module}/station_idx
4. Level 1 实例: misfit = EXTRACTORS[(operator, output)](intermediate, ctx) → /misfits/{name}
5. Level 2 实例（拓扑排序，bases 先算）: misfit = COMPOSERS[operator](...) → /misfits/{name}
```

## 9. use_misfit! 接口（shared/config）

```julia
# Level 1
Config.use_misfit!(:XcorrS, operator = Misfit.Xcorr, phase = "S",
                   output = Misfit.Xcorr.CC_MAX, channel = nothing)
# Level 2
Config.use_misfit!(:RelShift, operator = Aggregate.StdDev,
                   bases = [:AbsShiftP_Z, :AbsShiftS_N, :AbsShiftS_E],
                   output = Aggregate.StdDev.RELATIVE_OFFSET)
```

`use_misfit!` 注册时校验 `output ∈ operator.outputs()`；Level 1 实例 @eval 建实例 module
（使 `Config.{name}.trim() = ...` 覆盖生效）；Level 2 不建 module。参数覆盖沿用
`Config.{Instance}.trim() = [...]` 模式。

## 10. 兼容性

- `use_misfit!` 为 breaking change（`from::Symbol` → `operator::Module`），sample config 同步更新。
- C++ forward 必须按 (operator, phase, channel) 去重 kernel，避免同一计算重复跑。
- Level 2 依赖拓扑排序防循环；`best_lag` 为 Int32，CUDA 路径用 `int` 数组（`device.h`
  `parallel_for` 已支持多输出传参，无需改 backend）。
