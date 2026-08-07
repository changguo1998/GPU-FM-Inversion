# Design: Misfit 三层分解（Operator × Phase × Output）

## 1. 背景与动机

### 现状问题

当前 misfit 模块采用扁平命名（`XcorrP`、`XcorrS`、`PolarityP`），算子、震相、输出三者耦合：

- **算子与输出耦合**：`Xcorr.jl` 的 C++ kernel 内部同时算出 `max_abs_cc` 和对应的 `best_lag`，但只输出 `1.0 - max_abs_cc` 一个值，`best_lag`（时间偏移信息）被丢弃。无法复用同一份计算产出不同输出（如 AbsShift）。
- **无法表达组合关系**：`RelShift = StdDev(三分量 AbsShift)` 这类"消费已算好的基础 misfit 值"的组合无法表达。当前所有 misfit 都直接消费预处理数据。
- **C++ forward 硬编码**：`main.cpp` 硬编码 `/xcorrP`、`/xcorrS` 等组名，kernel 只产出单一 misfit 值写入 `/misfits/`。
- **输出选择无约束**：用户传 `:cc_max` 这类裸 Symbol，无编译期校验，易拼写错误。

### 设计目标

在保持 C++/GPU 性能的同时，获得 Julia 的灵活性：

1. **三层正交分解**：Misfit = Operator × Phase × Output，三者独立组合。
1. **C++/GPU 产出中间产物**：kernel 一次计算产出所有可用字段（cc_max + best_lag），不再丢弃信息。
1. **Julia 做语义解释**：Output extractor 从中间产物取值并变换；Composer 聚合基础 misfit 值。
1. **输出字段编译期安全**：`output = Xcorr.CC_MAX` 形式，IDE 可补全，拼写错误在注册时报错。

### 非目标

- 不改预处理阶段的数据流（`input.jl` 的 `process()` 逻辑不变，仍写 `database.h5`）。
- 不引入 ccall 共享库——C++/Julia 边界仍为 HDF5 文件交换。
- 不实现 PSR 的组合（PSR 算子已实现于 `shared/misfit/src/Psr.jl`，但 PSR 的组合/extractor 未实现，且 sample config 未注册 Psr 实例）。

## 2. 概念模型

```
Misfit = Operator × Phase × Output

  Operator   -> 算子（C++ kernel，GPU 加速），声明可用 outputs()
  Phase    -> 输入数据维度（P / S / 三分量组合）
  Output   -> 从 operator.outputs() 中选一个字段，由 Julia extractor 变换为 misfit 浮点数
```

### 两级 Misfit

**Level 1（Base）**：直接消费预处理数据，C++ kernel 产出中间产物。

```
XcorrP_CC  = XCorr(算子) + P(震相) + cc_max -> 1.0 - cc_max(输出)
AbsShiftP  = XCorr(算子) + P(震相) + best_lag -> best_lag * dt(输出)
```

同一 (Operator, Phase) 可派生多个 Base misfit，共享一次 kernel 运行，各自取不同字段。

**Level 2（Composed）**：消费 Level 1 已算好的 misfit 值，纯 Julia 聚合，不触及波形/GF 数据。

```
RelShift = StdDev(算子) + [AbsShiftP_Z, AbsShiftS_N, AbsShiftS_E](基础 misfit 集合) + relative_offset(输出)
```

Composer 按 station 对齐各 base misfit，跨分量取标准差。

## 3. 架构（B+C 混合）

```
┌─────────────────────────────────────────────────────────┐
│ C++ forward (GPU)                                       │
│   Kernel 产出 -> "中间产物" (raw intermediates)           │
│   写入 status_N.h5:/intermediates/                      │
│   不产出最终 misfit 值                                   │
└──────────────────────┬──────────────────────────────────┘
                       │ HDF5 文件交换
┌──────────────────────▼──────────────────────────────────┐
│ Julia assess (纯 CPU)                                   │
│   Output Extractor: 中间产物 -> misfit 浮点数             │
│   Composer: 基础 misfit 值 -> 组合 misfit 值              │
│   写入 status_N.h5:/misfits/                            │
└─────────────────────────────────────────────────────────┘
```

**C++ 只做重计算，Julia 做语义解释。**

### 为什么是 B+C

- **C 方案（纯 Julia）**：灵活但性能有限，放弃。
- **B 方案（输出提取器）**：kernel 输出中间产物，Julia 提取。灵活但需避免中间数据 I/O 瓶颈。
- **B+C 混合**：C++/GPU kernel 产出中间产物到 HDF5（中间数据量适中，见 §6），Julia 做 Output 提取与组合聚合。既保 GPU 性能，又获 Julia 灵活性。

## 4. 数据流

```
database.h5 (预处理数据: obs, gf, synamp, obs_norm2, pol_vec ...)
       │
       ▼
┌──────────────────────────────────────────────┐
│ C++ forward (GPU)                            │
│   按 (operator, phase) 去重，每组跑一次 kernel:   │
│                                              │
│   XCorr kernel (phase=P):                    │
│     输入: cc, synamp, obs_norm2, mt          │
│     输出: cc_max, best_lag  ← 中间产物         │
│                                              │
│   XCorr kernel (phase=S):                    │
│     输出: cc_max, best_lag                    │
│                                              │
│   Polarity kernel (phase=P):                 │
│     输出: syn_sign, dot_value                 │
└──────────┬───────────────────────────────────┘
           │ HDF5 write -> /intermediates/
           ▼
status_N.h5:/intermediates/
  XcorrP/cc_max      Float64[N_P × N_trials]
  XcorrP/best_lag    Int32[N_P × N_trials]
  XcorrS/cc_max      Float64[N_S × N_trials]
  XcorrS/best_lag    Int32[N_S × N_trials]
  PolarityP/syn_sign Int8[N_stations × N_trials]
  PolarityP/dot_value Float64[N_stations × N_trials]
           │ HDF5 read
           ▼
┌──────────────────────────────────────────────┐
│ Julia assess                                 │
│   Output Extractor (per instance):           │
│     XcorrP_CC:  cc_max -> 1.0 - cc_max       │
│     AbsShiftP:  best_lag -> best_lag * dt    │
│                                              │
│   Composer (per composed instance):          │
│     RelShift: std(shift_Z, shift_N, shift_E) per station
│                                              │
│   写入 /misfits/                              │
└──────────────────────────────────────────────┘
```

## 5. Operator Package 结构

将 `shared/misfit/` 从模板文件目录改为正式 Julia package，使输出字段常量在 `using` 后即可被 IDE 补全。

```
shared/misfit/
  Project.toml
  src/
    Misfit.jl              ← package 入口，导出各 operator 子模块
    Xcorr.jl               ← 模板体（常量 + 桩 + 预处理），无 module 包装
    Polarity.jl            ← 模板体（常量 + 桩 + 预处理），同上
```

### Xcorr.jl

模板文件只含函数体（无 `module` 包装），可被两处 `include` 复用：Misfit package 包装为 `module Xcorr`，`use_misfit!` 包装为实例 module。这样 `Misfit.Xcorr.CC_MAX` 可补全，实例的 `Config.XcorrP_CC.trim() = ...` 覆盖也生效。

```julia
# ── 输出字段常量（IDE 可补全，注册时校验）──
const CC_MAX   = :cc_max
const BEST_LAG = :best_lag

# ── 配置桩（实例可覆盖）──
function trim()::Vector{Float64}
    error("Xcorr.trim(): not implemented - return [-pre_sec, post_sec]")
end
function max_lag_periods()::Float64
    error("Xcorr.max_lag_periods(): not implemented")
end
function filter_order()::Int
    error("Xcorr.filter_order(): not implemented")
end
end
end
function band_low()::Vector{Int32}
    error("Xcorr.band_low(): not implemented")
end
function band_high()::Vector{Int32}
    error("Xcorr.band_high(): not implemented")
end

# ── Operator 元数据 ──
outputs() = [CC_MAX, BEST_LAG]
is_freq_dependent() = true

# ── 预处理逻辑（与现有 Xcorr.jl process() 相同，不变）──
function preprocess(obs, gf, dt, arrival_sample, low_cut, high_cut,
                    window_factor; filter_order = 4)
    # ... 现有实现 ...
end

function process(phases_pt, ptype, stations, picks, station_to_idx,
                 channel_data, gf_data, depths, low_cut, high_cut,
                 freq_idx, pf)
    # ... 现有实现 ...
end
```

Misfit package 入口包装模板：

```julia
# shared/misfit/src/Misfit.jl
module Misfit
    module Xcorr
        include("Xcorr.jl")
    end
    module Polarity
        include("Polarity.jl")
    end
end
```

### Polarity.jl

```julia
const SYN_SIGN  = :syn_sign
const DOT_VALUE = :dot_value

function trim()::Vector{Float64}
    error("Polarity.trim(): not implemented")
end

outputs() = [SYN_SIGN, DOT_VALUE]
is_freq_dependent() = false

function preprocess(gf, dt, arrival_sample, t_source, obs_polarity)
    # ... 现有实现 ...
end

function process(phases_pt, ptype, stations, picks, station_to_idx,
                 channel_data, gf_data, depths, pf, pol_f)
    # ... 现有实现 ...
end
```

### Composed operator（aggregate）

组合算子（如 `std_dev`）同样声明自己的 `outputs()`，供 Level 2 misfit 的 `output` 选择：

```julia
# shared/aggregate/src/StdDev.jl（新增 package 或并入 aggregate）
module StdDev
const RELATIVE_OFFSET = :relative_offset
const MEAN            = :mean
outputs() = [RELATIVE_OFFSET, MEAN]
end
```

## 6. Config 接口

### 新签名

```julia
# ── Level 1: Base misfit ──
Config.use_misfit!(
    name::Symbol;
    operator::Module,          # Misfit.Xcorr / Misfit.Polarity（模块引用）
    phase::String,           # "P" / "S"
    output::Symbol,          # operator.CC_MAX 等，须 ∈ operator.outputs()
    channel::Union{String, Nothing}=nothing,  # 可选：限定 channel（如 "Z"/"N"/"E"），见 §9
)

# ── Level 2: Composed misfit ──
Config.use_misfit!(
    name::Symbol;
    operator::Module,          # StdDev 等 aggregate operator 模块
    bases::Vector{Symbol},   # 基础 misfit 实例名，如 [:AbsShiftP_Z, :AbsShiftS_N, :AbsShiftS_E]
    output::Symbol,          # 须 ∈ operator.outputs()
)
```

### 注册时校验

`use_misfit!` 在创建实例前校验 `output ∈ operator.outputs()`，不在则报错：

```
ERROR: use_misfit!(:BadOne): output :syn_sign not in Xcorr.outputs() ([:cc_max, :best_lag])
```

### config.jl 用法

```julia
using Misfit            # 提供 Misfit.Xcorr, Misfit.Polarity
using Aggregate         # 提供 Aggregate.StdDev

# Level 1
Config.use_misfit!(:XcorrP_CC,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.CC_MAX)

Config.use_misfit!(:AbsShiftP,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.BEST_LAG)

Config.use_misfit!(:AbsShiftS_N,
    operator = Misfit.Xcorr, phase = "S", output = Misfit.Xcorr.BEST_LAG)
# 注：Z/N/E 区分由 channel 过滤实现，phase 仍为 "P"/"S", 详见 §9

Config.use_misfit!(:PolarityP,
    operator = Misfit.Polarity, phase = "P", output = Misfit.Polarity.SYN_SIGN)

# Level 2
Config.use_misfit!(:RelShift,
    operator = Aggregate.StdDev,
    bases  = [:AbsShiftP_Z, :AbsShiftS_N, :AbsShiftS_E],
    output = Aggregate.StdDev.RELATIVE_OFFSET)

# 参数覆盖（沿用现有模式）
Config.XcorrP_CC.trim()          = [-2.0, 5.0]
Config.XcorrP_CC.max_lag_periods() = 0.5
Config.AbsShiftP.trim()          = [-2.0, 5.0]   # 独立配置，即便共享 operator
```

### use_misfit! 内部逻辑

```julia
const _OPERATOR_MODULE = Dict{Symbol, Module}()   # name -> operator module
const _OUTPUT_FIELD  = Dict{Symbol, Symbol}()   # name -> output field
const _BASES         = Dict{Symbol, Vector{Symbol}}()  # composed name -> bases
const _IS_COMPOSED   = Set{Symbol}()
const _CHANNEL       = Dict{Symbol, String}()  # name -> channel filter (Level 1, optional)

function use_misfit!(name::Symbol; operator::Module, output::Symbol,
                     phase::Union{String, Nothing}=nothing, bases=nothing,
                     channel::Union{String, Nothing}=nothing)
    # 1. 校验 output ∈ operator.outputs()
    avail = operator.outputs()
    output ∈ avail ||
        error("use_misfit!($(name)): output $output not in $(nameof(operator)).outputs() ($avail)")

    # 2. Level 1 实例：include operator 模板文件创建实例 module
    #    （使 trim() 等函数定义在实例作用域内，Config.{name}.trim() = ... 覆盖才生效）
    #    Level 2（composed）无预处理函数、无 per-instance 覆盖，不创建 module
    if bases === nothing
        tmpl = _operator_template_path(operator)  # Misfit.Xcorr -> shared/misfit/src/Xcorr.jl
        @eval module $(name)
            include($(tmpl))
        end
    end

    # 3. 注册元数据
    _OPERATOR_MODULE[name] = operator
    _OUTPUT_FIELD[name]  = output
    if bases !== nothing
        _IS_COMPOSED = push!(_IS_COMPOSED, name)
        _BASES[name] = bases
    elseif phase !== nothing
        _PHASE_TYPE[name] = phase
        channel !== nothing && (_CHANNEL[name] = channel)
    end
    push!(_LOADED_MISFIT_MODULES, string(name))
    return nothing
end
```

## 7. HDF5 Schema 变更

### status_N.h5 新增 `/intermediates`

中间产物按 **(operator, phase)** 去重存储——共享同一 kernel 运行的多个实例共用一份中间产物。组名规则：`{OperatorName}{PhaseType}`（如 `XcorrP`、`XcorrS`、`PolarityP`）。

```
/intermediates/
  XcorrP/
    cc_max      Float64[N_phases_P × N_trials]    最大归一化 CC 值
    best_lag    Int32[N_phases_P × N_trials]      最佳偏移 lag 索引（0-based，相对 maxlag）
  XcorrS/
    cc_max      Float64[N_phases_S × N_trials]
    best_lag    Int32[N_phases_S × N_trials]
  PolarityP/
    syn_sign    Int8[N_stations × N_trials]       合成极性符号 (-1/0/1)
    dot_value   Float64[N_stations × N_trials]    原始点积值（置信度用）
```

`best_lag` 存相对偏移量（lag 索引 - maxlag），单位为采样点；Julia extractor 乘以 `dt` 转秒。

组名含 channel 后缀以区分三分量实例：无 channel 过滤时为 `{Operator}{Phase}`（如 `XcorrP`），有 channel 过滤时为 `{Operator}{Phase}_{channel}`（如 `XcorrS_H`、`XcorrS_V`）。相同 (operator, phase, channel) 的实例共享同一份中间产物（C++ forward 按 canonical key 去重 kernel 运行）。

### `/misfits` 保持不变

最终 misfit 值（extractor/composer 产出），按实例名存储：

```
/misfits/
  XcorrP_CC    Float64[N_phases_P × N_trials]
  AbsShiftP    Float64[N_phases_P × N_trials]
  RelShift     Float64[N_stations × N_trials]
  PolarityP    Float64[N_stations × N_trials]
```

### database.h5 `/config` 新增字段

`/config/{ModuleName}/` 增加元数据字段，供 C++ forward 决定跑哪些 kernel、Julia assess 决定如何 extract/compose：

```
/config/{ModuleName}/
  operator    String  scalar   "Xcorr" / "Polarity" / "StdDev"
  phase     String  scalar   "P" / "S"（Level 1）
  output    String  scalar   "cc_max" / "best_lag" / "relative_offset" ...
  channel   String  scalar   "Z" / "N" / "E" / "" （Level 1 可选，空表示不过滤）
  bases     String  [k]      基础 misfit 名（Level 2，Level 1 无此字段）
  is_composed Int8  scalar   0=base, 1=composed
```

## 8. 中间产物清单

各 operator kernel 产出的中间产物（C++ 侧定义，Julia 侧通过 `outputs()` 镜像）：

| Operator | 字段 | 类型 | 含义 |
|----------|-------------|---------|--------------------------------------------------|
| XCorr | `cc_max` | Float64 | 最大归一化互相关值（`max_k |cc_norm[k]|`） |
| XCorr | `best_lag` | Int32 | `cc_max` 对应的 lag 索引（相对 maxlag，单位采样点） |
| Polarity | `syn_sign` | Int8 | 合成极性符号（-1/0/1） |
| Polarity | `dot_value` | Float64 | 原始点积 `Σ pol_vec[i]·m[i]`（置信度加权用） |

### XCorr kernel 改造

现有 `xcorr_kernel.h` 内部循环已计算 `max_abs_cc`，只需在更新最大值时同时记录 `best_lag`：

```cpp
// 改造前（只输出 misfit）:
misfit[phase + trial * N_phases] = 1.0 - max_abs_cc;

// 改造后（输出两个中间产物）:
cc_max_out[phase + trial * N_phases]   = max_abs_cc;
best_lag_out[phase + trial * N_phases] = best_k - maxlag;  // 相对偏移
```

`best_k` 在现有 `if (abs_cc > max_abs_cc) { max_abs_cc = abs_cc; }` 处一并记录。性能开销可忽略。

## 9. Output Extractor 与 Composer

### Output Extractor（Level 1）

注册表形式，按 (operator, output_field) 查找变换函数：

```julia
# shared/aggregate/src/extractors.jl
const EXTRACTORS = Dict{Tuple{Symbol, Symbol}, Function}()

# XCorr 输出提取
EXTRACTORS[(:Xcorr, :cc_max)] =
    (intermediate, ctx) -> 1.0 .- intermediate["cc_max"]

EXTRACTORS[(:Xcorr, :best_lag)] =
    (intermediate, ctx) -> intermediate["best_lag"] .* ctx.dt   # 转秒

# Polarity 输出提取
EXTRACTORS[(:Polarity, :syn_sign)] =
    (intermediate, ctx) ->
        (intermediate["syn_sign"] .== ctx.obs_pol) ? 0.0 : 1.0  # 0=match,1=mismatch

EXTRACTORS[(:Polarity, :dot_value)] =
    (intermediate, ctx) -> abs.(intermediate["dot_value"])       # 置信度
```

`ctx` 携带 `dt`、`obs_pol`、`station_idx` 等 extractor 所需上下文（从 database.h5 读取）。

### Composer（Level 2）

组合算子按 operator 名注册聚合函数，输入为各 base misfit 对齐后的矩阵，输出为聚合 misfit 矩阵：

```julia
# shared/aggregate/src/composers.jl
const COMPOSERS = Dict{Symbol, Function}()

# StdDev: 跨基础 misfit 取标准差 + 均值，按 station 对齐
# 返回 Dict{Symbol, Matrix}，键为 StdDev.outputs() 中的字段
COMPOSERS[:std_dev] = function (base_misfits::Vector{Matrix{Float64}},
                                 base_station_idx::Vector{Vector{Int32}},
                                 ctx) -> Dict{Symbol, Matrix{Float64}}
    # base_misfits[i]: N_phases_i × N_trials（每个 base 的 misfit 值）
    # 按 station 聚合：每个 station 收集其各分量的值
    N_stations = ctx.N_stations
    N_trials   = size(base_misfits[1], 2)
    std_out = fill(NaN, N_stations, N_trials)
    mean_out = fill(NaN, N_stations, N_trials)
    for s in 1:N_stations, t in 1:N_trials
        vals = Float64[]
        for (i, m) in enumerate(base_misfits)
            for (p, si) in enumerate(base_station_idx[i])
                si == s && push!(vals, m[p, t])
            end
        end
        if length(vals) >= 2
            std_out[s, t]  = std(vals)
            mean_out[s, t] = mean(vals)
        end
    end
    return Dict(:relative_offset => std_out, :mean => mean_out)
end
```

### Composer 的 output 选择

组合算子也声明 `outputs()`，composer 产出多个聚合值，由 `output` 选择：

```julia
module StdDev
const RELATIVE_OFFSET = :relative_offset
const MEAN            = :mean
outputs() = [RELATIVE_OFFSET, MEAN]
end
```

`COMPOSERS[:std_dev]` 一次产出 `(relative_offset, mean)`，按实例的 `output` 字段取用。

### 三分量与 channel 子选择

RelShift 示例中 `AbsShiftP_Z`、`AbsShiftS_N`、`AbsShiftS_E` 是三个 Level 1 实例，对应同一台站三个分量的 AbsShift。当前 phase 模型中 phase key 含 channel（`{network}.{station}.{channel}.{phase_type}`），`use_misfit!` 的 `phase` 参数只选 phase_type（P/S），不区分 channel。

为支持三分量分别建实例，Level 1 增加可选 `channel` 过滤参数：

```julia
Config.use_misfit!(:AbsShiftS_N,
    operator = Misfit.Xcorr, phase = "S", channel = "N",
    output = Misfit.Xcorr.BEST_LAG)
```

`channel` 未指定时处理该 phase_type 的所有 channel（与现有行为一致）；指定时只处理匹配 channel 的 phase 条目。`input.jl` 的 `process()` 调用据此过滤 `phases_pt`。三分量实例的 `station_idx` 各自只含匹配 channel 的条目，Composer 按 station 对齐时跨实例聚合。

注：`channel` 是 Level 1 的可选正交维度，不影响 Operator×Phase×Output 核心模型；P 波通常单分量（Z），无需 channel 过滤。

## 10. assess.jl 流程

`scripts/assess.jl`（当前为空）实现 Julia 侧的 extract + compose：

```
1. 读 status_N.h5:/intermediates/  各 operator×phase 的中间产物
2. 读 database.h5:/config/{Module}/ 获取每个实例的 operator/phase/output/bases/is_composed
3. 读 database.h5:/station, /{Module}/station_idx 获取对齐上下文
4. 对每个 Level 1 实例:
     intermediate = read /intermediates/{Operator}{Phase}[_{channel}]/   # canonical key
     misfit = EXTRACTORS[(operator, output)](intermediate, ctx)
     write /misfits/{name}
5. 对每个 Level 2 实例（bases 依赖 Level 1，须拓扑排序）:
     base_misfits = [read /misfits/{b} for b in bases]
     misfit = COMPOSERS[operator](base_misfits, base_station_idx, ctx)
     write /misfits/{name}
```

拓扑排序：Level 2 实例的 `bases` 必须全部在 Level 1（或更早的 Level 2）中先算完。

## 11. 文件变更范围

| 文件 | 变更类型 | 说明 |
|-----------------------------------------|----------|------------------------------------------------------------------------------------------------|
| `shared/misfit/Project.toml` | 新增 | Misfit package 声明 |
| `shared/misfit/src/Misfit.jl` | 新增 | package 入口，`using` 导出 Xcorr/Polarity 子模块 |
| `shared/misfit/src/Xcorr.jl` | 改造 | 从模板文件改为 `module Xcorr`，加常量 CC_MAX/BEST_LAG + `outputs()` |
| `shared/misfit/src/Polarity.jl` | 改造 | 同上，加 SYN_SIGN/DOT_VALUE + `outputs()` |
| `shared/misfit/AGENTS.md` | 更新 | 反映 package 化 + 三层分解 |
| `shared/config/src/Config.jl` | 改造 | `use_misfit!` 新签名（operator::Module, output::Symbol, bases）+ 校验 + 元数据注册 |
| `shared/config/AGENTS.md` | 更新 | 新接口文档 |
| `shared/aggregate/Project.toml` | 新增 | Aggregate package 声明 |
| `shared/aggregate/src/Aggregate.jl` | 新增 | package 入口，导出 StdDev 等 |
| `shared/aggregate/src/StdDev.jl` | 新增 | `module StdDev`：常量 + `outputs()` |
| `shared/aggregate/src/extractors.jl` | 新增 | EXTRACTORS 注册表 |
| `shared/aggregate/src/composers.jl` | 新增 | COMPOSERS 注册表 |
| `shared/aggregate/AGENTS.md` | 新增 | Aggregate 模块文档 |
| `forward/src/kernels/xcorr_kernel.h` | 改造 | 加 `best_lag` 输出数组，不再输出 `1.0-cc_max` |
| `forward/src/kernels/polarity_kernel.h` | 改造 | 输出 `syn_sign` + `dot_value`，不再输出 0/1 misfit |
| `forward/src/main.cpp` | 改造 | kernel 输出中间产物到 `/intermediates/`；按 `/config` 的 (operator,phase,channel) 去重跑 kernel |
| `forward/src/data_cache.h` | 可能微调 | 缓存结构字段命名对齐 |
| `scripts/assess.jl` | 新增 | Julia 侧 extract + compose 主流程 |
| `scripts/input.jl` | 微调 | 适配 `use_misfit!` 新签名；写 `/config/{Module}/operator,phase,output,bases,is_composed` |
| `config_sample.jl` | 更新 | 示例新签名用法 |
| `doc/schema.md` | 更新 | 加 `/intermediates` schema + `/config` 新字段 |
| `doc/misfit-decomposition.md` | 新增 | 本设计文档 |

## 12. 实现阶段

建议分 4 个 commit，每个可独立验证：

1. **`feat(misfit): package 化 + 输出字段常量`**

   - `shared/misfit/` 转 package，Xcorr/Polarity 加 `outputs()` + 常量
   - `Config.use_misfit!` 新签名 + 校验
   - `config_sample.jl` 适配
   - 验证：`input.jl` 仍能跑通，产出 `database.h5` 含新 `/config` 字段

1. **`feat(forward): kernel 产出中间产物`**

   - `xcorr_kernel.h` 加 `best_lag` 输出
   - `polarity_kernel.h` 输出 `syn_sign` + `dot_value`
   - `main.cpp` 写 `/intermediates/` 而非 `/misfits/`
   - 验证：C++ forward 跑通，`status_N.h5` 含 `/intermediates/`

1. **`feat(aggregate): extractor + composer`**

   - `shared/aggregate/` 新 package
   - EXTRACTORS / COMPOSERS 注册表
   - `scripts/assess.jl` 实现 extract + compose
   - 验证：`assess.jl` 读 intermediates，产出 `/misfits/` 含 XcorrP_CC 等

1. **`feat(misfit): RelShift 组合示例`**

   - StdDev composer 实现
   - `config_sample.jl` 加 AbsShiftP_Z/S_N/S_E + RelShift 示例
   - `doc/schema.md` 更新
   - 验证：端到端跑通 AbsShift -> RelShift 组合

## 13. 兼容性与风险

- **向后兼容**：`use_misfit!` 签名 breaking change（从 `from::Symbol` 改为 `operator::Module`）。`config_sample.jl` 同步更新，无旧 config 需迁移。
- **中间产物 I/O**：`/intermediates/` 数据量 = 各 (operator,phase) 组的 N_phases × N_trials × 字段数。典型 100 phases × 100k trials × 2 字段 ≈ 160MB，HDF5 读写可接受。
- **kernel 去重**：C++ forward 须按 (operator, phase, channel) 去重跑 kernel，避免 XcorrP_CC 和 AbsShiftP 重复跑同一 XCorr-on-P 计算。去重依据 `/config` 中各实例的 operator+phase+channel。
- **拓扑排序**：Level 2 的 bases 引用须先于自身计算，assess.jl 实现拓扑排序防循环依赖。
- **GPU 路径**：`best_lag` 输出为 Int32，CUDA kernel 需用 `int` 数组；`device.h` 的 `parallel_for` 模板已支持多输出数组传参，无需改 backend 抽象。
