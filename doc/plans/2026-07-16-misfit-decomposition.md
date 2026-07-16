# Misfit 三层分解（Operator × Phase × Output）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 misfit 拆分为 Operator × Phase × Output 三层正交组合，C++/GPU kernel 产出中间产物，Julia extractor/composer 产出最终 misfit，支持 AbsShift/RelShift 组合。

**Architecture:** C++ forward kernel 产出中间产物（cc_max/best_lag/syn_sign/dot_value）写入 `status_N.h5:/intermediates/`；Julia assess 读中间产物，经 extractor 变换为 Level 1 misfit、经 composer 聚合为 Level 2 misfit，写入 `/misfits/`。Misfit 算子转为正式 Julia package，输出字段常量支持 IDE 补全与注册时校验。

**Tech Stack:** Julia 1.10+（shared packages）、C++17 + OpenMP/CUDA（forward kernels）、HDF5（数据交换）、Julia HDF5.jl / C++ libhdf5。

## Global Constraints

- 4-space indent，所有语言。Julia 走 `.JuliaFormatter.toml`（`indent=4`）。
- 所有角度 degrees，所有 HDF5 dataset `Float64` 除非注明；标量存标量 dataset。
- 改动文件在 staging 前跑 `bash format.sh`（覆盖 Julia/Markdown/C++）。
- 文档随代码更新：同次改动更新 `doc/` 与 per-module `AGENTS.md`。
- commit 用 conventional prefix：`feat:`/`fix:`/`refactor:`/`docs:`/`test:`/`build:`。
- HDF5 schema 是阶段间契约：`/intermediates` 新增需协调 forward 与 assess。
- phase key 格式 `{network}.{station}.{channel}.{phase_type}`；station_idx 1-based Int32。
- 模板文件保留 `Base.require(Base.PkgId(Base.UUID(...), "Signal"))` 解依赖机制（在 package module 与动态 instance module 两种上下文均可用，无需声明 dep）。
- 不提交 git 除非用户明确要求；计划内 commit 步骤供执行时遵循。

设计依据：`doc/misfit-decomposition.md`（完整 spec）、`doc/schema.md`（HDF5 schema）。

______________________________________________________________________

## File Structure

| 文件 | 责任 | 任务 |
|-----------------------------------------|----------------------------------------------------------------------------|-------|
| `shared/misfit/Project.toml` | Misfit package 声明（deps: Signal, IO） | T1 |
| `shared/misfit/src/Misfit.jl` | package 入口，包装 Xcorr/Polarity 为子 module | T1 |
| `shared/misfit/src/Xcorr.jl` | 模板体：CC_MAX/BEST_LAG 常量 + outputs() + 桩 + preprocess/process | T1 |
| `shared/misfit/src/Polarity.jl` | 模板体：SYN_SIGN/DOT_VALUE 常量 + outputs() + 桩 + preprocess/process | T1 |
| `shared/misfit/test/runtests.jl` | outputs() 与常量可访问性测试 | T1 |
| `shared/config/src/Config.jl` | `use_misfit!` 新签名 + 校验 + 元数据注册 | T2 |
| `shared/config/Project.toml` | 加 Signal/IO dep（instance 模板 include 后可用） | T2 |
| `shared/config/test/runtests.jl` | 注册/校验/元数据测试 | T2 |
| `forward/src/kernels/xcorr_kernel.h` | 加 `best_lag` 输出数组 | T3 |
| `forward/src/kernels/polarity_kernel.h` | 输出 `syn_sign`+`dot_value` | T3 |
| `forward/src/main.cpp` | 写 `/intermediates/`，按 (operator,phase,channel) 去重 | T3 |
| `forward/src/data_cache.h` | （可能微调）字段命名对齐 | T3 |
| `shared/aggregate/Project.toml` | Aggregate package 声明（UUID `d899c1fe-439a-47d9-a1ba-d896c8e97e6b`，已预留） | T4 |
| `shared/aggregate/src/Aggregate.jl` | package 入口，导出 StdDev + EXTRACTORS/COMPOSERS | T4 |
| `shared/aggregate/src/StdDev.jl` | `module StdDev`：RELATIVE_OFFSET/MEAN 常量 + outputs() | T4 |
| `shared/aggregate/src/extractors.jl` | EXTRACTORS 注册表 | T4 |
| `shared/aggregate/src/composers.jl` | COMPOSERS 注册表 | T4 |
| `shared/aggregate/test/runtests.jl` | extractor/composer 测试 | T4 |
| `scripts/assess.jl` | 读 intermediates -> extract -> compose -> 写 misfits | T4 |
| `config_sample.jl` | 新签名示例 + AbsShift/RelShift | T2,T4 |
| `Project.toml` | 加 `Misfit` dep + path | T1 |
| `doc/schema.md` | 加 `/intermediates` + `/config` 新字段 | T4 |
| `shared/misfit/AGENTS.md` | 反映 package 化 + 三层分解 | T1 |
| `shared/config/AGENTS.md` | 新 use_misfit! 接口 | T2 |
| `shared/aggregate/AGENTS.md` | Aggregate 模块文档 | T4 |

______________________________________________________________________

## Task 1: Misfit package 化 + 输出字段常量

**Files:**

- Create: `shared/misfit/Project.toml`
- Create: `shared/misfit/src/Misfit.jl`
- Create: `shared/misfit/src/Xcorr.jl`（从现 `shared/misfit/Xcorr.jl` 迁移 + 加常量/outputs）
- Create: `shared/misfit/src/Polarity.jl`（从现 `shared/misfit/Polarity.jl` 迁移 + 加常量/outputs）
- Create: `shared/misfit/test/runtests.jl`
- Modify: `Project.toml`（加 Misfit dep + path）
- Modify: `shared/misfit/AGENTS.md`

**Interfaces:**

- Produces: `Misfit.Xcorr`、`Misfit.Polarity` 模块；常量 `Misfit.Xcorr.CC_MAX`/`BEST_LAG`、`Misfit.Polarity.SYN_SIGN`/`DOT_VALUE`；函数 `outputs()` 返回 `Vector{Symbol}`。

- 模板体（preprocess/process）签名不变，供 T2 的 `use_misfit!` include 复用。

- [ ] **Step 1: 写失败测试**

Create `shared/misfit/test/runtests.jl`:

```julia
using Misfit
using Test

@testset "Xcorr outputs" begin
    @test :cc_max in Misfit.Xcorr.outputs()
    @test :best_lag in Misfit.Xcorr.outputs()
    @test Misfit.Xcorr.CC_MAX == :cc_max
    @test Misfit.Xcorr.BEST_LAG == :best_lag
    @test Misfit.Xcorr.is_freq_dependent() == true
end

@testset "Polarity outputs" begin
    @test :syn_sign in Misfit.Polarity.outputs()
    @test :dot_value in Misfit.Polarity.outputs()
    @test Misfit.Polarity.SYN_SIGN == :syn_sign
    @test Misfit.Polarity.DOT_VALUE == :dot_value
    @test Misfit.Polarity.is_freq_dependent() == false
end
```

- [ ] **Step 2: 跑测试确认失败**

Run: `julia --project=. -e 'using Pkg; Pkg.test("Misfit")'`
Expected: FAIL（`Misfit` 未注册/不存在）

- [ ] **Step 3: 创建 package 声明**

Create `shared/misfit/Project.toml`:

```toml
name = "Misfit"
uuid = "1a03ced9-fb88-487f-aa94-9f0e83bb8bf2"
version = "0.1.0"

[deps]
IO = "4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"
Signal = "c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"
```

- [ ] **Step 4: 创建 package 入口**

Create `shared/misfit/src/Misfit.jl`:

```julia
module Misfit

module Xcorr
    include("Xcorr.jl")
end

module Polarity
    include("Polarity.jl")
end

end # module Misfit
```

- [ ] **Step 5: 迁移 Xcorr 模板**

Create `shared/misfit/src/Xcorr.jl`。从现 `shared/misfit/Xcorr.jl` 复制全部内容，在文件顶部（export 之前）插入常量与 outputs()，并在 export 列表后加 `outputs`。具体：保留现有 `_Signal`/`_IO` 的 `Base.require(PkgId(...))` 依赖机制、所有 `function` 桩与 `preprocess`/`process` 实现不变。仅在 export 区上方插入：

```julia
# ── 输出字段常量（IDE 可补全，注册时校验）──
const CC_MAX   = :cc_max
const BEST_LAG = :best_lag

# ── Operator 元数据 ──
outputs() = [CC_MAX, BEST_LAG]
```

并将 `export` 行扩展为：

```julia
export trim, maxlag_factor, filter_order, outputs
export select_threshold,
    deselect_threshold, preprocess, process, is_freq_dependent, band_low, band_high
```

（`is_freq_dependent() = true` 已存在，保留。）

- [ ] **Step 6: 迁移 Polarity 模板**

Create `shared/misfit/src/Polarity.jl`。从现 `shared/misfit/Polarity.jl` 复制全部，顶部插入：

```julia
# ── 输出字段常量（IDE 可补全，注册时校验）──
const SYN_SIGN  = :syn_sign
const DOT_VALUE = :dot_value

# ── Operator 元数据 ──
outputs() = [SYN_SIGN, DOT_VALUE]
```

`export` 行改为：

```julia
export trim, preprocess, process, is_freq_dependent, outputs
```

（`is_freq_dependent() = false` 已存在，保留。）

- [ ] **Step 7: 注册到 root Project.toml**

Modify `Project.toml`，在 `[deps]` 加一行，`[paths]` 加一行：

```toml
[deps]
# ... existing ...
Misfit = "1a03ced9-fb88-487f-aa94-9f0e83bb8bf2"

[paths]
# ... existing ...
Misfit = "shared/misfit"
```

- [ ] **Step 8: 跑测试确认通过**

Run: `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.test("Misfit")'`
Expected: PASS（两组 testset 全绿）

- [ ] **Step 9: 删除旧模板文件**

现 `shared/misfit/Xcorr.jl` 与 `Polarity.jl` 内容已迁入 `src/`，删除根目录旧文件（保留 `AGENTS.md`）。`shared/misfit/AGENTS.md` 的 `## Files` 表更新为 `src/Xcorr.jl`、`src/Polarity.jl`、`src/Misfit.jl`。

- [ ] **Step 10: 跑格式 + input.jl 冒烟**

Run: `bash format.sh --check`（应全绿）；`julia --project=. scripts/input.jl config_sample.jl`（此时 `config_sample.jl` 仍用旧 `from=:Xcorr` 签名，会因 T2 改 Config 而失败--T1 阶段先注释掉 config_sample 的 use_misfit! 行验证 package 加载，或在 T2 一并改）。

> 注：T1 不改 `Config.use_misfit!`，故 `config_sample.jl` 的 `use_misfit!` 调用仍指向旧机制。T1 验收以 `Pkg.test("Misfit")` 通过为准；input.jl 端到端在 T2 改完 Config 与 config_sample 后验证。

- [ ] **Step 11: Commit**

```bash
git add shared/misfit/ Project.toml
git commit -m "feat(misfit): package 化 + 输出字段常量 outputs()"
```

______________________________________________________________________

## Task 2: Config.use_misfit! 新签名 + 校验

**Files:**

- Modify: `shared/config/src/Config.jl`（`use_misfit!` 重写 + 元数据注册 + 新 accessor）
- Modify: `shared/config/Project.toml`（加 Signal/IO/Misfit dep）
- Modify: `config_sample.jl`（新签名）
- Create: `shared/config/test/runtests.jl`
- Modify: `shared/config/AGENTS.md`

**Interfaces:**

- Consumes: T1 的 `Misfit.Xcorr`/`Misfit.Polarity`（`outputs()`、常量）。

- Produces:

  - `use_misfit!(name; operator::Module, output::Symbol, phase::Union{String,Nothing}=nothing, bases=nothing, channel::Union{String,Nothing}=nothing)`
  - accessor `operator_module(name)::Module`、`output_field(name)::Symbol`、`bases_of(name)`、`is_composed(name)::Bool`、`channel_of(name)`、`phase_type(name)`（已有，保留）
  - instance module `Config.{name}`（Level 1，含模板函数可覆盖）

- [ ] **Step 1: 写失败测试**

Create `shared/config/test/runtests.jl`:

```julia
using Config
using Misfit
using Test

@testset "use_misfit! base registration" begin
    Config.use_misfit!(:T2_XcorrP,
        operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.CC_MAX)
    @test Config.phase_type(:T2_XcorrP) == "P"
    @test Config.output_field(:T2_XcorrP) == :cc_max
    @test Config.operator_module(:T2_XcorrP) === Misfit.Xcorr
    @test !Config.is_composed(:T2_XcorrP)
    @test Config.channel_of(:T2_XcorrP) === nothing
    @test "T2_XcorrP" in Config.misfit_modules()
    # instance module 可覆盖
    Config.T2_XcorrP.trim() = [-2.0, 5.0]
    @test Config.T2_XcorrP.trim() == [-2.0, 5.0]
end

@testset "use_misfit! validation" begin
    @test_throws Exception Config.use_misfit!(:T2_Bad,
        operator = Misfit.Xcorr, phase = "P", output = :nonexistent)
end

@testset "use_misfit! composed registration" begin
    Config.use_misfit!(:T2_Rel,
        operator = Misfit.Xcorr,  # 占位，T4 换 StdDev
        bases = [:T2_XcorrP], output = Misfit.Xcorr.CC_MAX)
    @test Config.is_composed(:T2_Rel)
    @test Config.bases_of(:T2_Rel) == [:T2_XcorrP]
    @test !isdefined(Config, :T2_Rel)  # Level 2 不创建 instance module
end

@testset "channel filter" begin
    Config.use_misfit!(:T2_XcorrSH,
        operator = Misfit.Xcorr, phase = "S", channel = "H",
        output = Misfit.Xcorr.BEST_LAG)
    @test Config.channel_of(:T2_XcorrSH) == "H"
end
```

- [ ] **Step 2: 跑测试确认失败**

Run: `julia --project=. -e 'using Pkg; Pkg.test("Config")'`
Expected: FAIL（`use_misfit!` 旧签名不匹配，accessor 不存在）

- [ ] **Step 3: 改 Config 包依赖**

Modify `shared/config/Project.toml`:

```toml
name = "Config"
uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
version = "0.1.0"

[deps]
IO = "4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"
Misfit = "1a03ced9-fb88-487f-aa94-9f0e83bb8bf2"
Signal = "c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"
```

- [ ] **Step 4: 重写 use_misfit! 与元数据**

Modify `shared/config/src/Config.jl`。替换现有 `_MISFIT_DIR`/`_LOADED_MISFIT_MODULES`/`_PHASE_TYPE`/`use_misfit!`/`phase_type` 整块（约第 36-99 行），改为：

```julia
# Misfit operator plugin loader

const _MISFIT_DIR = joinpath(@__DIR__, "..", "..", "misfit", "src")
const _LOADED_MISFIT_MODULES = String[]
const _PHASE_TYPE = Dict{Symbol, String}()
const _OPERATOR_MODULE = Dict{Symbol, Module}()   # name -> operator module
const _OUTPUT_FIELD = Dict{Symbol, Symbol}()      # name -> output field
const _BASES = Dict{Symbol, Vector{Symbol}}()     # composed name -> bases
const _IS_COMPOSED = Set{Symbol}()
const _CHANNEL = Dict{Symbol, String}()           # name -> channel filter (Level 1, optional)

# operator module -> template file path
_operator_template_path(op::Module) =
    joinpath(_MISFIT_DIR, "$(nameof(op)).jl")

"""
    use_misfit!(name; operator, output, phase=nothing, bases=nothing, channel=nothing)

Register a misfit instance. Level 1 (base): `operator` + `phase` + `output`.
Level 2 (composed): `operator` (aggregate) + `bases` + `output`.

`output` must be in `operator.outputs()`. Level 1 creates `Config.{name}` instance
module (per-instance parameter overrides via `Config.{name}.trim() = ...`).
Level 2 does not create an instance module.
"""
function use_misfit!(
    name::Symbol;
    operator::Module,
    output::Symbol,
    phase::Union{String, Nothing} = nothing,
    bases = nothing,
    channel::Union{String, Nothing} = nothing,
)
    avail = operator.outputs()
    output ∈ avail ||
        error("use_misfit!($(name)): output $output not in $(nameof(operator)).outputs() ($avail)")

    if bases === nothing
        # Level 1: include operator template into instance module
        tmpl = _operator_template_path(operator)
        @eval module $(name)
            include($(tmpl))
        end
        _PHASE_TYPE[name] = phase
        channel !== nothing && (_CHANNEL[name] = channel)
    else
        _IS_COMPOSED = push!(_IS_COMPOSED, name)
        _BASES[name] = bases
    end

    _OPERATOR_MODULE[name] = operator
    _OUTPUT_FIELD[name] = output
    n = string(name)
    !(n in _LOADED_MISFIT_MODULES) && push!(_LOADED_MISFIT_MODULES, n)
    return nothing
end

# Accessors
operator_module(name::Symbol)::Module = _OPERATOR_MODULE[name]
output_field(name::Symbol)::Symbol = _OUTPUT_FIELD[name]
bases_of(name::Symbol) = _BASES[name]
is_composed(name::Symbol)::Bool = name in _IS_COMPOSED
channel_of(name::Symbol)::Union{String, Nothing} = get(_CHANNEL, name, nothing)

"""
    phase_type(name::Symbol) -> Union{String, Nothing}
"""
function phase_type(name::Symbol)::Union{String, Nothing}
    return get(_PHASE_TYPE, name, nothing)
end
```

保留 `misfit_modules()`（返回 `copy(_LOADED_MISFIT_MODULES)`）不变。删除旧 `from::Symbol` 签名与 `_PHASE_TYPE` 旧注释块。

- [ ] **Step 5: 跑测试确认通过**

Run: `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.test("Config")'`
Expected: PASS（4 个 testset 全绿）

- [ ] **Step 6: 更新 config_sample.jl**

Modify `config_sample.jl`，替换顶部 use_misfit! 块为：

```julia
using Misfit

Config.use_misfit!(:XcorrP,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.CC_MAX)
Config.use_misfit!(:XcorrS,
    operator = Misfit.Xcorr, phase = "S", output = Misfit.Xcorr.CC_MAX)
Config.use_misfit!(:PolarityP,
    operator = Misfit.Polarity, phase = "P", output = Misfit.Polarity.SYN_SIGN)
```

（参数覆盖 `Config.XcorrP.trim() = [-2.0, 5.0]` 等不变。）

- [ ] **Step 7: input.jl 写 /config 新字段**

Modify `scripts/input.jl` 第 307-319 行 `db_config` 构建块，为每个实例写 `operator`/`output`/`phase`/`channel`/`is_composed`/`bases`。将：

```julia
for (m_name, mod) in module_instances
    cfg_entry = Dict{String, Any}("trim" => Float64.(mod.trim()))
    if isdefined(mod, :maxlag_factor)
        ...
    end
    db_config[m_name] = cfg_entry
end
```

先加固 `module_instances` 构建循环（约 input.jl:200-204，对每个 module 调 `getfield(Config, sym)`）。composed 实例无 instance module，必须跳过，否则 `getfield` 崩溃。将：

```julia
module_instances = Dict{String, Module}()
for m_name in misfit_modules
    mod = getfield(Config, Symbol(m_name))
    module_instances[m_name] = mod
end
```

改为：

```julia
module_instances = Dict{String, Module}()
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) && continue   # Level 2 无 instance module
    module_instances[m_name] = getfield(Config, sym)
end
```

再改 `db_config` 构建块（约 input.jl:307-319）。Level 1 遍历 `module_instances`（已只含 Level 1），Level 2 单独遍历 composed：

```julia
# Level 1: 有 instance module 的实例
for (m_name, mod) in module_instances
    sym = Symbol(m_name)
    cfg_entry = Dict{String, Any}("trim" => Float64.(mod.trim()))
    if isdefined(mod, :maxlag_factor)
        cfg_entry["maxlag_factor"] = Float64(mod.maxlag_factor())
        cfg_entry["filter_order"] = Int32(mod.filter_order())
        cfg_entry["select_threshold"] = Float64(mod.select_threshold())
        cfg_entry["deselect_threshold"] = Float64(mod.deselect_threshold())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    cfg_entry["operator"] = string(nameof(Config.operator_module(sym)))
    cfg_entry["output"] = string(Config.output_field(sym))
    cfg_entry["is_composed"] = Int8(0)
    cfg_entry["phase"] = Config.phase_type(sym)
    ch = Config.channel_of(sym)
    cfg_entry["channel"] = ch === nothing ? "" : ch
    db_config[m_name] = cfg_entry
end

# Level 2: composed 实例（无 instance module、无预处理参数）
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) || continue
    db_config[m_name] = Dict{String, Any}(
        "operator" => string(nameof(Config.operator_module(sym))),
        "output" => string(Config.output_field(sym)),
        "is_composed" => Int8(1),
        "bases" => String.(Config.bases_of(sym)),
    )
end
```

> 注：composed 实例无 `trim` 等参数，`/config` 只写 `operator`/`output`/`is_composed`/`bases`。T2 阶段 config_sample 无 composed 实例，第二循环空跑；T4 注册 RelShift 后生效。

- [ ] **Step 8: 端到端冒烟**

Run: `bash format.sh` && `julia --project=. scripts/input.jl config_sample.jl`
Expected: input.jl 跑通，`database.h5:/config/XcorrP` 含 `operator`/`output`/`phase`/`channel`/`is_composed` 字段。用 `julia --project=. -e 'using IO; println(IO.read_config("database.h5"))'` 验证（路径以实际 data_dir 为准）。

- [ ] **Step 9: 更新 AGENTS.md + Commit**

更新 `shared/config/AGENTS.md` 的 `use_misfit!` 行与新 accessor 表。然后：

```bash
git add shared/config/ config_sample.jl scripts/input.jl
git commit -m "feat(config): use_misfit! 新签名 operator/output + 校验"
```

______________________________________________________________________

## Task 3: forward kernel 产出中间产物

**Files:**

- Modify: `forward/src/kernels/xcorr_kernel.h`（加 `best_lag` 输出）
- Modify: `forward/src/kernels/polarity_kernel.h`（输出 `syn_sign`+`dot_value`）
- Modify: `forward/src/main.cpp`（写 `/intermediates/`，去重 kernel）
- Modify: `forward/src/data_cache.h`（如需字段对齐）

**Interfaces:**

- Consumes: `database.h5` 预处理数据（不变）；`/config/{Module}` 的 `operator`/`phase`/`channel` 字段（T2 写入）。
- Produces: `status_N.h5:/intermediates/{Operator}{Phase}[_{channel}]/{field}`，字段：XCorr→`cc_max`(Float64)+`best_lag`(Int32)；Polarity→`syn_sign`(Int8)+`dot_value`(Float64)。

> **测试约束**：forward 为 C++ 无单元测试框架。验收靠运行二进制 + 检查 HDF5 输出（项目 AGENTS.md：Tests run manually）。每个 kernel 改动附最小数值验证（手算单点）。

- [ ] **Step 1: xcorr kernel 加 best_lag 输出**

Modify `forward/src/kernels/xcorr_kernel.h`。函数签名加 `int32_t *best_lag_out` 参数，删除 `misfit` 输出。改造 `launch_xcorr_misfit` 签名与循环体：

```cpp
template <Backend B>
inline void
launch_xcorr_misfit(const double *mt,
                    const double *cc_data,
                    const double *synamp_data,
                    const double *obs_norm2,
                    double *cc_max_out,        // [N_phases × N_trials] 中间产物
                    int32_t *best_lag_out,     // [N_phases × N_trials] 相对 maxlag
                    int N_phases, int N_trials, int cc_pp, int maxlag) {
    Device<B>::parallel_for(N_phases * N_trials, [=](int idx) {
        const int phase = idx / N_trials;
        const int trial = idx % N_trials;

        double m[6];
        for (int c = 0; c < 6; ++c)
            m[c] = mt[trial + c * N_trials];

        double syn_norm2 = 0.0;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                syn_norm2 += m[i] * synamp_data[phase + (i * 6 + j) * N_phases] * m[j];

        const double obs_n2 = obs_norm2[phase];
        if (syn_norm2 <= 0.0 || obs_n2 <= 0.0) {
            cc_max_out[phase + trial * N_phases] = 0.0;
            best_lag_out[phase + trial * N_phases] = 0;
            return;
        }

        const double denom = std::sqrt(obs_n2 * syn_norm2);
        const int cc_start = phase * cc_pp;
        double max_abs_cc = 0.0;
        int best_k = maxlag;  // 默认零偏移
        for (int k = 0; k < cc_pp; ++k) {
            double cc_syn = 0.0;
            for (int i = 0; i < 6; ++i)
                cc_syn += m[i] * cc_data[(cc_start + k) + i * (N_phases * cc_pp)];
            double cc_norm = cc_syn / denom;
            double abs_cc = std::fabs(cc_norm);
            if (abs_cc > max_abs_cc) {
                max_abs_cc = abs_cc;
                best_k = k;
            }
        }
        cc_max_out[phase + trial * N_phases] = max_abs_cc;
        best_lag_out[phase + trial * N_phases] = static_cast<int32_t>(best_k - maxlag);
    });
}
```

- [ ] **Step 2: polarity kernel 输出 syn_sign + dot_value**

Modify `forward/src/kernels/polarity_kernel.h`。签名改为输出两个中间产物：

```cpp
template <Backend B>
void launch_polarity_kernel(
    const double *mt, const double *pol_vec, const double *obs_pol,
    int8_t *syn_sign_out,    // [N_stations × N_trials] -1/0/1
    double *dot_value_out,   // [N_stations × N_trials] 原始点积
    int N_stations, int N_trials) {
    Device<B>::parallel_for(N_stations * N_trials, [=](int idx) {
        const int station = idx / N_trials;
        const int trial = idx % N_trials;
        double obs = obs_pol[station];
        if (std::isnan(obs)) {
            syn_sign_out[station + trial * N_stations] = 0;
            dot_value_out[station + trial * N_stations] = NAN;
            return;
        }
        double dot = 0.0;
        for (int c = 0; c < 6; ++c)
            dot += pol_vec[station + c * N_stations] * mt[trial + c * N_trials];
        int syn = (dot > 0.0) ? 1 : ((dot < 0.0) ? -1 : 0);
        syn_sign_out[station + trial * N_stations] = static_cast<int8_t>(syn);
        dot_value_out[station + trial * N_stations] = dot;
    });
}
```

- [ ] **Step 3: main.cpp 写 /intermediates/**

Modify `forward/src/main.cpp`。核心改动：

1. 读 `/config/misfit_modules` 与各 `/config/{Module}/` 的 `operator`/`phase`/`channel`，构建去重 key 集合 `std::set<tuple<string,string,string>>`（operator, phase, channel）。
1. 为每个去重 key 分配中间产物输出数组（XCorr: cc_max+best_lag；Polarity: syn_sign+dot_value）。
1. kernel 调用改用新签名，输出中间产物数组。
1. 写入 `/intermediates/{key}/` 组（key 形如 `XcorrP`、`XcorrS_H`）。

替换第 6 节 kernel 调用与第 7 节写入（约 main.cpp:151-321）。XCorr 写入段示例：

```cpp
// 写 /intermediates/XcorrP/cc_max, /intermediates/XcorrP/best_lag
if (!status_file.group_exists("/intermediates"))
    status_file.create_group("/intermediates");
std::string ig = "/intermediates/" + op_name + phase;  // e.g. "XcorrP"
if (channel_filter != "")
    ig += "_" + channel_filter;
if (!status_file.group_exists(ig))
    status_file.create_group(ig);
status_file.write_double_2d(ig + "/cc_max", cc_max_out.data(),
                            (hsize_t)N_phases, (hsize_t)N_trials);
status_file.write_int_2d(ig + "/best_lag", best_lag_out.data(),
                         (hsize_t)N_phases, (hsize_t)N_trials);
```

Polarity 写入 `syn_sign`（Int8 2D）与 `dot_value`（Double 2D）。删除旧 `/misfits/xcorr`、`/misfits/polarity`、`/misfits/psr` 写入（PSR 暂留中间产物或延后）。

> `data_cache.h` 的 `CacheEntry` 无需改结构（输入侧不变）。`hdf5_io.h` 若无 `write_int_2d`（Int32）与 Int8 写入，需补充--检查 `hdf5_io.cpp` 现有写入方法，缺则按 `write_double_2d` 模板加 `write_int32_2d` 与 `write_int8_2d`。

- [ ] **Step 4: 构建 + 运行验收**

Run: `cd forward && cmake -B build && cmake --build build`
然后：`./build/forward <database.h5> <status_N.h5>`（需先有 trials，可用 input.jl 产出的 status_0.h5 + 手造 trials，或等 Phase 2 preprocess.jl；T3 阶段可用 `tests/synthetic_data.jl` 造数据）。
Expected: `status_N.h5` 含 `/intermediates/XcorrP/cc_max`、`/intermediates/XcorrP/best_lag`、`/intermediates/PolarityP/syn_sign`、`/intermediates/PolarityP/dot_value`。用 `h5dump -H status_N.h5` 或 Julia `IO.read_config` 验证字段存在与 shape。

- [ ] **Step 5: 数值手算验证（单点）**

取 N_trials=1、N_phases=1、固定 MT，手算 cc_max 与 best_lag，对比 HDF5 值。记录验证过程到 commit message 或 `doc/stages/forward.md`（当前为空，填充）。

- [ ] **Step 6: Commit**

```bash
git add forward/ doc/stages/forward.md
git commit -m "feat(forward): kernel 产出中间产物到 /intermediates/"
```

______________________________________________________________________

## Task 4: aggregate extractor + composer + assess.jl + RelShift

**Files:**

- Create: `shared/aggregate/Project.toml`
- Create: `shared/aggregate/src/Aggregate.jl`
- Create: `shared/aggregate/src/StdDev.jl`
- Create: `shared/aggregate/src/extractors.jl`
- Create: `shared/aggregate/src/composers.jl`
- Create: `shared/aggregate/test/runtests.jl`
- Create: `scripts/assess.jl`
- Modify: `config_sample.jl`（加 AbsShift + RelShift 示例）
- Modify: `doc/schema.md`（加 `/intermediates` + `/config` 新字段）
- Create: `shared/aggregate/AGENTS.md`

**Interfaces:**

- Consumes: T2 的 Config 元数据（`operator_module`/`output_field`/`bases_of`/`is_composed`）；T3 的 `/intermediates/`；`IO.read_config`/`read_misfits`/`read_stations`。

- Produces: `status_N.h5:/misfits/{name}` 最终 misfit 矩阵；`Aggregate.StdDev` 模块；`EXTRACTORS`/`COMPOSERS` 注册表。

- [ ] **Step 1: 写 extractor 失败测试**

Create `shared/aggregate/test/runtests.jl`:

```julia
using Aggregate
using Test

@testset "StdDev outputs" begin
    @test :relative_offset in Aggregate.StdDev.outputs()
    @test :mean in Aggregate.StdDev.outputs()
end

@testset "XCorr extractors" begin
    inter = Dict("cc_max" => [0.8 0.6; 0.9 0.5], "best_lag" => Int32[2 -1; 0 3])
    ctx = (dt = 0.5,)
    cc = Aggregate.EXTRACTORS[(:Xcorr, :cc_max)](inter, ctx)
    @test cc ≈ [0.2 0.4; 0.1 0.5]
    sh = Aggregate.EXTRACTORS[(:Xcorr, :best_lag)](inter, ctx)
    @test sh ≈ [1.0 -0.5; 0.0 1.5]
end

@testset "Polarity extractors" begin
    inter = Dict("syn_sign" => Int8[1 -1; 1 1], "dot_value" => [0.5 -0.3; 1.2 0.0])
    ctx = (obs_pol = Int8[1 -1; 1 -1],)
    m = Aggregate.EXTRACTORS[(:Polarity, :syn_sign)](inter, ctx)
    @test m ≈ [0.0 0.0; 0.0 1.0]  # match=0, mismatch=1
end

@testset "StdDev composer" begin
    # 两 base，每 base 1 phase × 2 trial，同属 station 1
    base_misfits = [[1.0 2.0], [3.0 4.0]]
    base_station_idx = [Int32[1], Int32[1]]
    ctx = (N_stations = 1,)
    out = Aggregate.COMPOSERS[:std_dev](base_misfits, base_station_idx, ctx)
    @test :relative_offset in keys(out)
    @test out[:relative_offset][1, 1] ≈ std([1.0, 3.0])
    @test out[:relative_offset][1, 2] ≈ std([2.0, 4.0])
end
```

- [ ] **Step 2: 跑测试确认失败**

Run: `julia --project=. -e 'using Pkg; Pkg.test("Aggregate")'`
Expected: FAIL（Aggregate 不存在）

- [ ] **Step 3: 创建 Aggregate package**

Create `shared/aggregate/Project.toml`（UUID 用 root Project.toml 已预留的）:

```toml
name = "Aggregate"
uuid = "d899c1fe-439a-47d9-a1ba-d896c8e97e6b"
version = "0.1.0"

[deps]
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

Create `shared/aggregate/src/Aggregate.jl`:

```julia
module Aggregate

using Statistics

export EXTRACTORS, COMPOSERS

include("StdDev.jl")
include("extractors.jl")
include("composers.jl")

end # module Aggregate
```

- [ ] **Step 4: StdDev 模块**

Create `shared/aggregate/src/StdDev.jl`:

```julia
module StdDev
const RELATIVE_OFFSET = :relative_offset
const MEAN = :mean
outputs() = [RELATIVE_OFFSET, MEAN]
end # module StdDev
```

- [ ] **Step 5: extractors 注册表**

Create `shared/aggregate/src/extractors.jl`:

```julia
const EXTRACTORS = Dict{Tuple{Symbol, Symbol}, Function}()

# XCorr: cc_max -> 1 - cc_max
EXTRACTORS[(:Xcorr, :cc_max)] =
    (inter, ctx) -> 1.0 .- inter["cc_max"]

# XCorr: best_lag -> best_lag * dt (秒)
EXTRACTORS[(:Xcorr, :best_lag)] =
    (inter, ctx) -> Float64.(inter["best_lag"]) .* ctx.dt

# Polarity: syn_sign -> 0/1 mismatch
EXTRACTORS[(:Polarity, :syn_sign)] =
    (inter, ctx) -> Float64.(
        Int8.(inter["syn_sign"]) .!= ctx.obs_pol)

# Polarity: dot_value -> |dot| (置信度)
EXTRACTORS[(:Polarity, :dot_value)] =
    (inter, ctx) -> abs.(inter["dot_value"])
```

- [ ] **Step 6: composers 注册表**

Create `shared/aggregate/src/composers.jl`:

```julia
const COMPOSERS = Dict{Symbol, Function}()

# StdDev: 跨 base misfit 取 std + mean，按 station 对齐
# base_misfits[i]: Matrix{Float64} (N_phases_i × N_trials)
# base_station_idx[i]: Vector{Int32} (每 phase 的 station_idx)
# 返回 Dict{Symbol, Matrix{Float64}}，键为 StdDev.outputs()
COMPOSERS[:std_dev] = function (base_misfits::Vector{Matrix{Float64}},
                                 base_station_idx::Vector{Vector{Int32}},
                                 ctx)
    N_stations = ctx.N_stations
    N_trials = size(base_misfits[1], 2)
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
            std_out[s, t] = std(vals)
            mean_out[s, t] = mean(vals)
        end
    end
    return Dict(:relative_offset => std_out, :mean => mean_out)
end
```

- [ ] **Step 7: 跑测试确认通过**

Run: `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.test("Aggregate")'`
Expected: PASS（4 个 testset 全绿）

- [ ] **Step 8: 写 assess.jl**

Create `scripts/assess.jl`（flat script，顶层直列）:

```julia
#!/usr/bin/env julia
# assess.jl - 读 intermediates -> extract -> compose -> 写 misfits
# Usage: julia scripts/assess.jl <database.h5> <status_N.h5>

using HDF5
using IO, Config, Aggregate

db_path = ARGS[1]
status_path = ARGS[2]

# 1. 读 /config 元数据
cfg = IO.read_config(db_path)
modules = cfg["misfit_modules"]

# 2. 读 station 对齐上下文
stations = IO.read_stations(db_path)
N_stations = length(stations)

# 读 trials 维度
trials = IO.read_trials(status_path)
N_trials = length(trials.strike)

# 3. Level 1: extract
misfits = Dict{String, Matrix{Float64}}()
level2 = String[]  # 延后处理
for m_name in modules
    mcfg = cfg[m_name]
    is_composed = Int8(mcfg["is_composed"]) == 1
    if is_composed
        push!(level2, m_name)
        continue
    end
    op = Symbol(mcfg["operator"])
    out = Symbol(mcfg["output"])
    phase = mcfg["phase"]
    ch = mcfg["channel"]
    # canonical intermediate group key
    key = string(op, phase)
    ch != "" && (key = key * "_" * ch)
    grp = "/intermediates/$key"
    inter = h5open(f -> begin
        d = Dict{String, Any}()
        for k in keys(f[grp])
            d[k] = read(f[grp][k])
        end
        d
    end, status_path, "r")
    # 构造 ctx
    ctx = if op == :Xcorr
        s0 = stations[1]
        (dt = s0.dt,)
    elseif op == :Polarity
        picks = IO.read_phase_picks(db_path)
        sta_to_pick = Dict(p.station_id => p for p in picks)
        obs_pol = Int8[get(sta_to_pick, s.id, IO.PhasePick("", "", "", Int8(-128))).P_polarity
                       for s in stations]
        (obs_pol = obs_pol,)
    else
        (;)
    end
    misfits[m_name] = EXTRACTORS[(op, out)](inter, ctx)
end

# 4. Level 2: compose（拓扑：bases 必须已算）
# 简化：反复扫描直到所有 level2 完成
remaining = copy(level2)
while !isempty(remaining)
    progressed = false
    for m_name in collect(remaining)
        mcfg = cfg[m_name]
        bs = String.(mcfg["bases"])
        if all(b in keys(misfits) for b in bs)
            op = Symbol(mcfg["operator"])
            out = Symbol(mcfg["output"])
            base_misfits = [misfits[b] for b in bs]
            base_station_idx = [Int32.(cfg[b]["station_idx"]) for b in bs]
            ctx = (N_stations = N_stations,)
            res = COMPOSERS[op](base_misfits, base_station_idx, ctx)
            misfits[m_name] = res[out]
            deleteat!(remaining, findfirst(==(m_name), remaining))
            progressed = true
        end
    end
    progressed || error("assess: circular or unresolved bases in $remaining")
end

# 5. 写 /misfits/
h5open(status_path, "r+") do f
    !haskey(f, "misfits") && create_group(f, "misfits")
    for (m_name, m) in misfits
        haskey(f["misfits"], m_name) && delete_object(f["misfits"], m_name)
        f["misfits"][m_name] = m
    end
end

@info "assess: wrote $(length(misfits)) misfit matrices to $status_path"
```

> 注：`cfg[b]["station_idx"]` 需 T2 在 `/config/{Module}` 写入 station_idx（或 assess 从 `/{Module}/station_idx` 读，已在 database.h5 per-module 组中）。确认 IO.read_config 是否递归读到 per-module 的 station_idx；若 /config 不含，改读 `database.h5:/{Module}/station_idx`。

- [ ] **Step 9: config_sample.jl 加 AbsShift + RelShift**

Modify `config_sample.jl`，在现有 use_misfit! 后加：

```julia
using Aggregate

Config.use_misfit!(:AbsShiftP,
    operator = Misfit.Xcorr, phase = "P", output = Misfit.Xcorr.BEST_LAG)
Config.use_misfit!(:AbsShiftSH,
    operator = Misfit.Xcorr, phase = "S", channel = "H", output = Misfit.Xcorr.BEST_LAG)
Config.use_misfit!(:AbsShiftSV,
    operator = Misfit.Xcorr, phase = "S", channel = "V", output = Misfit.Xcorr.BEST_LAG)
Config.use_misfit!(:RelShift,
    operator = Aggregate.StdDev,
    bases = [:AbsShiftP, :AbsShiftSH, :AbsShiftSV],
    output = Aggregate.StdDev.RELATIVE_OFFSET)
```

- [ ] **Step 10: 更新 doc/schema.md**

在 `doc/schema.md` 的 `status_{N}.h5` 节加 `/intermediates` 子节，在 `/config` 节加 `operator`/`output`/`phase`/`channel`/`bases`/`is_composed` 字段表（内容取自 `doc/misfit-decomposition.md` §7）。

- [ ] **Step 11: 端到端验收**

Run: `bash format.sh` && 全链路：`input.jl` -> `forward` -> `assess.jl`
Expected: `status_N.h5:/misfits/` 含 `XcorrP`/`AbsShiftP`/`RelShift`/`PolarityP`。验证 RelShift shape 为 `[N_stations × N_trials]`，AbsShift 为 `[N_phases × N_trials]`。

- [ ] **Step 12: Commit**

```bash
git add shared/aggregate/ scripts/assess.jl config_sample.jl doc/schema.md
git commit -m "feat(aggregate): extractor + composer + assess.jl + RelShift"
```

______________________________________________________________________

## Self-Review

**1. Spec coverage**（对照 `doc/misfit-decomposition.md`）：

- §2 概念模型 Operator×Phase×Output → T1（outputs/常量）+T2（签名）✓
- §3 B+C 架构 C++/intermediates + Julia/extract+compose → T3+T4 ✓
- §5 Method package → T1 ✓
- §6 Config 接口 → T2 ✓
- §7 /intermediates schema → T3（写）+T4（schema.md）✓
- §8 中间产物清单 cc_max/best_lag/syn_sign/dot_value → T3 ✓
- §9 extractor/composer → T4 ✓
- §10 assess.jl 流程 → T4 ✓
- §9 channel 子选择 → T2（channel 参数）+T3（去重 key）✓

**2. Placeholder scan**：无 TBD/TODO；process() 体以"从现文件复制+精确行号"指示，非占位。

**3. Type consistency**：`operator`/`output` 命名贯穿 T1-T4；`EXTRACTORS[(:Xcorr,:cc_max)]` 与 `outputs()` 常量一致；`COMPOSERS[:std_dev]` 返回 Dict 与 `StdDev.outputs()` 一致；`best_lag` Int32 与 §7 一致。

**已知风险**（计划内标注，非占位）：

- T3 C++ 无单测框架，验收靠运行+HDF5 检查+手算（AGENTS.md 允许手动测试）。
- T4 assess.jl 的 `station_idx` 读取路径需确认 `/config` vs per-module 组（Step 8 已标注排查点）。
- T3 `hdf5_io.h` 可能缺 Int32/Int8 写入方法（Step 3 已标注补充点）。
