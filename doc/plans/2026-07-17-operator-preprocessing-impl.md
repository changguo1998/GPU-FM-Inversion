# 算子预处理重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 `doc/plans/2026-07-17-operator-preprocessing.md` 设计:共享预处理层 + 算子 reductions 分层,per-lag Xcorr,PSR 接入,三分量 N/E/Z,删 select/deselect。

**Architecture:** input.jl 分两层 -- Layer 0 共享预处理(完整波形 去均值/去趋势/taper/bandpass,持久化中间结果);Layer 1 算子 reductions(消费已预处理波形)。算子 process 接收已预处理波形,只算 reductions。Xcorr 改 per-lag synamp(lag)/dot_obs_gf(lag)。Polarity 用 source_duration()。PSR 新建接入。三分量 N/E/Z 不旋转。

**Tech Stack:** Julia (HDF5.jl, DSP.jl, LinearAlgebra);C++ forward kernel(后续消费 reductions,本计划不涉及)。

## Global Constraints

- 4 空格缩进,JuliaFormatter `indent=4`,空格绕运算符(AGENTS.md)。
- 角度 degrees,HDF5 datasets `Float64` 除非注明(AGENTS.md)。
- `bash format.sh --check` 必须通过(staging 前);`bash format.sh` 格式化。
- docs follow code:代码改动的同次更新 `doc/` + per-module `AGENTS.md`(AGENTS.md)。
- 不主动 commit:除非用户明确要求,不 `git commit`(AGENTS.md)。task gate = `format.sh --check` + 测试通过。
- flat scripts:stage scripts 顶层直列,允许私有自包含辅助函数(AGENTS.md)。
- conventional commits(若用户要求 commit):`feat:`/`fix:`/`refactor:`/`docs:` 前缀。
- caveman 通信(实现交互),但代码/注释英文(AGENTS.md)。

______________________________________________________________________

## File Structure

| 文件 | 职责 | 动作 |
|------------------------------------------------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `shared/signal/src/Signal.jl` | 共享预处理原子函数 + `preprocess_waveform!` | 改:新增 demean/detrend/taper/preprocess_waveform!,改 trim_time_window! 非对称,删旧 preprocess_xcorr!/polarity!/psr! |
| `shared/signal/test/runtests.jl` | Signal 单测 | 新建 |
| `shared/misfit/src/Xcorr.jl` | Xcorr 算子(per-lag reductions) | 改:trim 周期数,max_lag_periods,删 select/deselect,preprocess 改 per-lag,process 接预处理波形 |
| `shared/misfit/src/Polarity.jl` | Polarity 算子 | 改:source_duration,process 接预处理波形 |
| `shared/misfit/src/Psr.jl` | PSR 算子 | 新建 |
| `shared/misfit/src/Misfit.jl` | 算子注册 | 改:注册 Psr |
| `shared/misfit/test/runtests.jl` | Misfit 单测 | 改:扩展 Xcorr/Polarity/PSR 测试 |
| `scripts/input.jl` | 数据接入 + 预处理 | 改:Layer 0 + Layer 1 + 持久化 + 参数写入 |
| `config_sample.jl`、`examples/synthetic/config.jl` | 配置 | 改:参数名 + channel 标签 |
| `doc/schema.md`、`doc/misfit-decomposition.md`、`doc/modules/waveform_proc.md` | 文档 | 改 |

______________________________________________________________________

### Task 1: Signal 共享预处理层

**Files:**

- Modify: `shared/signal/src/Signal.jl`
- Create: `shared/signal/test/runtests.jl`

**Interfaces:**

- Produces: `Signal.demean!(wf::Vector{Float64})`, `Signal.detrend!(wf::Vector{Float64})`, `Signal.taper!(wf::Vector{Float64}; frac::Float64=0.05)`, `Signal.preprocess_waveform!(wf, dt, low_cut, high_cut; demean=true, detrend=true, taper=true, order=4, do_bandpass=true) -> wf_proc`, `Signal.trim_time_window!(obs, gf, dt, arrival, pre_periods, post_periods, band_high) -> (obs_trim, gf_trim)` (非对称)

- Consumes: 无(原子层)

- [ ] **Step 1: 写失败测试** `shared/signal/test/runtests.jl`

```julia
using Signal
using Test

@testset "demean/detrend/taper" begin
    wf = [1.0, 2.0, 3.0, 4.0, 5.0]
    Signal.demean!(wf)
    @test maximum(abs.(wf)) < 1e-9                   # 均值约 0
    wf2 = [1.0, 3.0, 5.0, 7.0, 9.0]                  # 线性趋势
    Signal.detrend!(wf2)
    @test maximum(abs.(wf2)) < 1e-9                  # 去趋势后约 0
    wf3 = ones(100)
    Signal.taper!(wf3; frac = 0.1)
    @test wf3[1] < 1.0 && wf3[50] ≈ 1.0 && wf3[end] < 1.0
end

@testset "preprocess_waveform!" begin
    dt = 0.1
    wf = randn(500)
    low_cut, high_cut = 0.1, 0.5
    out = Signal.preprocess_waveform!(copy(wf), dt, low_cut, high_cut)
    @test length(out) == length(wf)                  # 完整波形,不裁窗
    @test !isnan.(out) |> all
    # do_bandpass=false 只清理不滤波
    out2 = Signal.preprocess_waveform!(copy(wf), dt, low_cut, high_cut; do_bandpass = false)
    @test length(out2) == length(wf)
end

@testset "trim_time_window! non-symmetric" begin
    dt = 0.1
    obs = collect(1.0:100.0)
    gf = reshape(collect(1.0:600.0), 100, 6)
    arrival = 50
    # pre_periods=2, post_periods=5, band_high=0.5 -> pre_sec=4, post_sec=10
    ot, gt = Signal.trim_time_window!(obs, gf, dt, arrival, 2.0, 5.0, 0.5)
    pre_n = round(Int, 2.0 / 0.5 / dt)               # 40
    post_n = round(Int, 5.0 / 0.5 / dt)              # 100
    @test size(ot, 1) == pre_n + post_n + 1
    @test ot[1] ≈ obs[arrival - pre_n]
    @test ot[end] ≈ obs[arrival + post_n]
end
```

- [ ] **Step 2: 跑测试验证失败**

Run: `julia --project=shared/signal shared/signal/test/runtests.jl`
Expected: FAIL (`demean!` not defined / `trim_time_window!` 签名不匹配)

- [ ] **Step 3: 实现 Signal.jl 新函数**

在 `shared/signal/src/Signal.jl` 中:

```julia
export demean!, detrend!, taper!, preprocess_waveform!

"""Remove mean from waveform in-place."""
function demean!(wf::Vector{Float64})
    wf .-= sum(wf) / length(wf)
    return wf
end

"""Remove linear trend (least-squares) from waveform in-place."""
function detrend!(wf::Vector{Float64})
    n = length(wf)
    t = collect(1.0:n)
    # least squares: a + b*t
    s_tt = sum(abs2, t) - (sum(t)^2) / n
    s_ty = dot(t, wf) - (sum(t) * sum(wf)) / n
    b = s_ty / s_tt
    a = (sum(wf) - b * sum(t)) / n
    wf .-= (a .+ b .* t)
    return wf
end

"""Apply cosine taper to both ends in-place."""
function taper!(wf::Vector{Float64}; frac::Float64 = 0.05)
    n = length(wf)
    ntap = max(1, round(Int, frac * n))
    for i in 1:ntap
        w = 0.5 * (1 - cos(pi * (i - 1) / ntap))
        wf[i] *= w
        wf[n - i + 1] *= w
    end
    return wf
end

"""
    preprocess_waveform!(wf, dt, low_cut, high_cut;
                         demean=true, detrend=true, taper=true, order=4, do_bandpass=true)
                         -> wf_proc

Full-waveform preprocessing: demean -> detrend -> taper -> bandpass.
Returns processed full waveform (no trimming). do_bandpass=false skips filter
(for Polarity, which is not freq-dependent).
"""
function preprocess_waveform!(
    wf::Vector{Float64},
    dt::Float64,
    low_cut::Float64,
    high_cut::Float64;
    demean::Bool = true,
    detrend::Bool = true,
    taper::Bool = true,
    order::Int = 4,
    do_bandpass::Bool = true,
)
    demean && demean!(wf)
    detrend && detrend!(wf)
    taper && taper!(wf)
    do_bandpass && bandpass_filter!(wf, dt, low_cut, high_cut; order = order)
    return wf
end
```

改 `trim_time_window!` 为非对称(替换原对称版本):

```julia
"""
    trim_time_window!(obs, gf, dt, arrival_sample, pre_periods, post_periods, band_high)
                      -> (obs_trimmed, gf_trimmed)

Non-symmetric trim: [arrival - pre_periods/band_high, arrival + post_periods/band_high].
pre_periods/post_periods are dimensionless period counts (window scales with band).
"""
function trim_time_window!(
    obs::Vector{Float64},
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    pre_periods::Float64,
    post_periods::Float64,
    band_high::Float64,
)
    pre_sec = pre_periods / band_high
    post_sec = post_periods / band_high
    pre_n = max(1, round(Int, pre_sec / dt))
    post_n = max(1, round(Int, post_sec / dt))
    start_idx = max(1, arrival_sample - pre_n)
    end_idx = min(length(obs), arrival_sample + post_n)
    return obs[start_idx:end_idx], gf[start_idx:end_idx, :]
end
```

删除 `preprocess_xcorr!`、`preprocess_polarity!`、`preprocess_psr!` 三个旧函数及其 export。保留 `trim_to_polarity_window!`(Polarity 仍用)、`bandpass_filter!`、`rms_amplitude`、`envelope`。

- [ ] **Step 4: 跑测试验证通过**

Run: `julia --project=shared/signal shared/signal/test/runtests.jl`
Expected: PASS

- [ ] **Step 5: format check**

Run: `bash format.sh --check`
Expected: 无改动

______________________________________________________________________

### Task 2: Xcorr per-lag reductions

**Files:**

- Modify: `shared/misfit/src/Xcorr.jl`
- Modify: `shared/misfit/test/runtests.jl`

**Interfaces:**

- Produces: `Xcorr.trim()` (周期数), `Xcorr.max_lag_periods()`, `Xcorr.preprocess(gf_proc_full, obs_win, dt, arrival, pre_periods, post_periods, band_high, max_lag_periods) -> (obs_norm2, synamp_lag, dot_obs_gf_lag)`, `Xcorr.process(...)` (接已预处理波形)

- Consumes: `Signal.preprocess_waveform!`, `Signal.trim_time_window!`

- [ ] **Step 1: 写失败测试** (扩展 `shared/misfit/test/runtests.jl`)

```julia
@testset "Xcorr params" begin
    @test :cc_max in Misfit.Xcorr.outputs()
    @test :best_lag in Misfit.Xcorr.outputs()
    @test Misfit.Xcorr.is_freq_dependent() == true
    # select/deselect deleted
    @test !isdefined(Misfit.Xcorr, :select_threshold)
    @test !isdefined(Misfit.Xcorr, :deselect_threshold)
    @test !isdefined(Misfit.Xcorr, :maxlag_factor)
    @test isdefined(Misfit.Xcorr, :max_lag_periods)
end

@testset "Xcorr per-lag reductions" begin
    # minimal synthetic: obs window + full gf, 1 depth, 1 band
    dt = 0.1; arrival = 50; pre_p = 2.0; post_p = 5.0; band_high = 0.5
    max_lag_p = 3.0
    obs_win = randn(141)                 # (2/0.5 + 5/0.5)/0.1 + 1
    gf_full = randn(200, 6)
    obs_n2, synamp_lag, dog_lag = Misfit.Xcorr.preprocess(
        gf_full, obs_win, dt, arrival, pre_p, post_p, band_high, max_lag_p)
    L = size(synamp_lag, 3)
    @test L == 2 * round(Int, max_lag_p / band_high / dt) + 1
    @test size(synamp_lag) == (6, 6, L)
    @test size(dog_lag) == (6, L)
    @test obs_n2 ≈ dot(obs_win, obs_win)
    # synamp symmetric
    @test synamp_lag[:, :, 1] ≈ synamp_lag[:, :, 1]'
end
```

- [ ] **Step 2: 跑测试验证失败**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: FAIL (`max_lag_periods` not defined / `preprocess` 签名不匹配)

- [ ] **Step 3: 实现 Xcorr.jl**

Config stubs 段:删 `maxlag_factor`/`select_threshold`/`deselect_threshold`,新增 `max_lag_periods`:

```julia
function max_lag_periods()::Float64
    error("Xcorr.max_lag_periods(): not implemented - return Float64 period count (e.g. 3.0)")
end
```

改 `trim()` 注释为周期数:`return [-pre_periods, post_periods] (e.g. [-2.0, 5.0])`。

改 export 行:删 `maxlag_factor, select_threshold, deselect_threshold`,加 `max_lag_periods`。

重写 `preprocess`(per-lag reductions,接已预处理 GF 完整波形 + obs 固定窗):

```julia
"""
    preprocess(gf_full, obs_win, dt, arrival_sample, pre_periods, post_periods,
               band_high, max_lag_periods)
               -> (obs_norm2, synamp_lag, dot_obs_gf_lag)

Compute per-lag reductions for cross-correlation.
- obs_win: fixed trimmed obs window [arrival-pre, arrival+post] (already preprocessed)
- gf_full: full preprocessed GF waveform [N_full, 6]
- synamp_lag[l] = gf_full[win-l]' * gf_full[win-l]  (6x6)
- dot_obs_gf_lag[l] = obs_win' * gf_full[win-l]     (6-vector)
- obs_norm2 = obs_win' * obs_win
"""
function preprocess(
    gf_full::Matrix{Float64},
    obs_win::Vector{Float64},
    dt::Float64,
    arrival_sample::Int,
    pre_periods::Float64,
    post_periods::Float64,
    band_high::Float64,
    max_lag_periods::Float64,
)
    pre_sec = pre_periods / band_high
    post_sec = post_periods / band_high
    pre_n = max(1, round(Int, pre_sec / dt))
    post_n = max(1, round(Int, post_sec / dt))
    nt_win = pre_n + post_n + 1
    max_lag_sec = max_lag_periods / band_high
    max_lag_n = min(round(Int, max_lag_sec / dt), (nt_win - 1) ÷ 2)
    L = 2 * max_lag_n + 1

    obs_norm2 = dot(obs_win, obs_win)
    synamp_lag = zeros(Float64, 6, 6, L)
    dog_lag = zeros(Float64, 6, L)

    # window in full-gf coordinates: [arrival-pre_n, arrival+post_n]
    w_start = arrival_sample - pre_n
    for (li, lag) in enumerate(-max_lag_n:max_lag_n)
        s = w_start - lag                     # syn full window start
        e = s + nt_win - 1
        if s < 1 || e > size(gf_full, 1)
            continue                          # lag out of range, leave zeros
        end
        gf_sub = gf_full[s:e, :]
        synamp_lag[:, :, li] = gf_sub' * gf_sub
        dog_lag[:, li] = gf_sub' * obs_win
    end
    return obs_norm2, synamp_lag, dog_lag
end
```

`process` 函数改:接收已预处理 obs(完整)+ GF(完整),内部先 `trim_time_window!` 取 obs 固定窗,再调 `preprocess` 算 per-lag reductions。返回 Dict 含 `obs_norm2`、`synamp_lag`、`dot_obs_gf_lag`、`obs`(裁窗,调试)、`gf`(裁窗 lag=0,调试)。参数从 `maxlag_factor()` 改 `max_lag_periods()`。删 select/deselect 相关写入。

(注:`process` 主体结构保留两遍 pass:Pass1 收集 + 定 L,Pass2 填充。`low_cut/high_cut` 参数移除 -- 滤波已在 Layer 0 完成,process 接已滤波波形。`freq_idx` 保留用于 per-band 持久化。)

- [ ] **Step 4: 跑测试验证通过**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: PASS

- [ ] **Step 5: format check**

Run: `bash format.sh --check`
Expected: 无改动

______________________________________________________________________

### Task 3: Polarity source_duration 分离

**Files:**

- Modify: `shared/misfit/src/Polarity.jl`
- Modify: `shared/misfit/test/runtests.jl`

**Interfaces:**

- Produces: `Polarity.source_duration()` (秒), `Polarity.preprocess(gf_full, dt, arrival, source_duration) -> gf_pol` (接已基础清理 GF)

- Consumes: `Signal.trim_to_polarity_window!`

- [ ] **Step 1: 写失败测试**

```julia
@testset "Polarity params" begin
    @test :syn_sign in Misfit.Polarity.outputs()
    @test :dot_value in Misfit.Polarity.outputs()
    @test Misfit.Polarity.is_freq_dependent() == false
    @test !isdefined(Misfit.Polarity, :trim)
    @test isdefined(Misfit.Polarity, :source_duration)
end
```

- [ ] **Step 2: 跑测试验证失败**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: FAIL (`source_duration` not defined, `trim` still defined)

- [ ] **Step 3: 实现 Polarity.jl**

Config stubs:删 `trim()`,新增:

```julia
function source_duration()::Float64
    error("Polarity.source_duration(): not implemented - return Float64 seconds (e.g. 2.0)")
end
```

改 export:`trim` -> `source_duration`。

`preprocess` 改(接已基础清理 GF,只裁 polarity 窗):

```julia
function preprocess(gf_full::Matrix{Float64}, dt::Float64, arrival_sample::Int,
                    source_duration::Float64)
    return _Signal.trim_to_polarity_window!(gf_full, dt, arrival_sample, source_duration)
end
```

`process`:`t_source = source_duration()`(替 `trim()[2]`)。obs_pol 逻辑不变。返回 Dict 不变(`obs_pol`、`gf_pol`)。

- [ ] **Step 4: 跑测试验证通过**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: PASS

- [ ] **Step 5: format check**

Run: `bash format.sh --check`

______________________________________________________________________

### Task 4: PSR 算子新建

**Files:**

- Create: `shared/misfit/src/Psr.jl`
- Modify: `shared/misfit/src/Misfit.jl`
- Modify: `shared/misfit/test/runtests.jl`

**Interfaces:**

- Produces: `Misfit.Psr` 模块,`Psr.is_freq_dependent()=true`,`Psr.outputs()=[:psr_value]`,`Psr.PSR_VALUE=:psr_value`,`Psr.pre_P()/post_P()/pre_S()/post_S()` (周期数),`Psr.preprocess(...)` ,`Psr.process(...)`

- Consumes: `Signal.preprocess_waveform!`, `Signal.rms_amplitude`

- [ ] **Step 1: 写失败测试**

```julia
@testset "PSR outputs" begin
    @test :psr_value in Misfit.Psr.outputs()
    @test Misfit.Psr.PSR_VALUE == :psr_value
    @test Misfit.Psr.is_freq_dependent() == true
    @test isdefined(Misfit.Psr, :pre_P)
    @test isdefined(Misfit.Psr, :post_S)
end
```

- [ ] **Step 2: 跑测试验证失败**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: FAIL (`Misfit.Psr` not defined)

- [ ] **Step 3: 实现 Psr.jl**

`shared/misfit/src/Psr.jl` (仿 Xcorr.jl/Polarity.jl 结构):

```julia
# PSR misfit plugin (template)
# Included inside Config.{name} (dynamically created inner module).
# Template for P/S amplitude ratio - instantiated per phase pair via
# Config.use_misfit!(:Psr, operator = Misfit.Psr, ...).

const PSR_VALUE = :psr_value

outputs() = [PSR_VALUE]
is_freq_dependent() = true

export pre_P, post_P, pre_S, post_S, preprocess, process, outputs, is_freq_dependent, PSR_VALUE

# -- Config namespace (user must override) --
function pre_P()::Float64
    error("Psr.pre_P(): not implemented - return period count (e.g. 1.0)")
end
function post_P()::Float64
    error("Psr.post_P(): not implemented - return period count (e.g. 3.0)")
end
function pre_S()::Float64
    error("Psr.pre_S(): not implemented - return period count (e.g. 1.0)")
end
function post_S()::Float64
    error("Psr.post_S(): not implemented - return period count (e.g. 5.0)")
end
```

`preprocess`(接已预处理 P/S GF 完整波形 + obs,算 rms reductions):

```julia
"""
    preprocess(gf_P_full, gf_S_full, obs_P_full, obs_S_full, dt,
               arrival_P, arrival_S, pre_P, post_P, pre_S, post_S, band_high)
               -> (amp_P, amp_S, obs_psr)

amp_P = GF_P[win]' * GF_P[win] (6x6), amp_S likewise.
obs_psr = log10(rms(obs_P_win) / rms(obs_S_win)).
"""
function preprocess(gf_P_full, gf_S_full, obs_P_full, obs_S_full, dt,
                    arrival_P, arrival_S, pre_P_p, post_P_p, pre_S_p, post_S_p, band_high)
    # trim windows (period counts -> seconds via band_high)
    p_pre_n = max(1, round(Int, pre_P_p / band_high / dt))
    p_post_n = max(1, round(Int, post_P_p / band_high / dt))
    s_pre_n = max(1, round(Int, pre_S_p / band_high / dt))
    s_post_n = max(1, round(Int, post_S_p / band_high / dt))
    p_s = max(1, arrival_P - p_pre_n); p_e = min(length(obs_P_full), arrival_P + p_post_n)
    s_s = max(1, arrival_S - s_pre_n); s_e = min(length(obs_S_full), arrival_S + s_post_n)
    amp_P = gf_P_full[p_s:p_e, :]' * gf_P_full[p_s:p_e, :]
    amp_S = gf_S_full[s_s:s_e, :]' * gf_S_full[s_s:s_e, :]
    obs_psr = _Signal.rms_amplitude(obs_P_full[p_s:p_e]) / _Signal.rms_amplitude(obs_S_full[s_s:s_e])
    obs_psr = obs_psr > 0 ? log10(obs_psr) : 0.0
    return amp_P, amp_S, obs_psr
end
```

`process` 仿 Xcorr.jl:遍历 phases(P/S pair per station),调 `preprocess`,返回 Dict(`amp_P`、`amp_S`、`obs_psr`,per band×depth)。需 P/S phase 配对逻辑(同 station 的 P arrival + S arrival)。

`Misfit.jl` 加:

```julia
module Psr
include("Psr.jl")
end
```

- [ ] **Step 4: 跑测试验证通过**

Run: `julia --project=shared/misfit shared/misfit/test/runtests.jl`
Expected: PASS

- [ ] **Step 5: format check**

Run: `bash format.sh --check`

______________________________________________________________________

### Task 5: input.jl Layer 0 + Layer 1 重构

**Files:**

- Modify: `scripts/input.jl`

**Interfaces:**

- Consumes: `Signal.preprocess_waveform!`, 各算子 `process`

- Produces: database.h5 含 Layer 0 中间结果 + Layer 1 reductions

- [ ] **Step 1: 写 e2e 验证脚本** `tmp/verify_input.jl`

```julia
# Run input.jl on synthetic, check database.h5 has new schema groups
include("scripts/input.jl")
using HDF5
db = "examples/synthetic/database.h5"
h5open(db, "r") do f
    @assert exists(f, "preprocess")          # Layer 0 obs filtered
    @assert exists(f, "gf_preprocessed")     # Layer 0 gf filtered
    # Xcorr per-lag
    for mn in keys(f)
        occursin("Xcorr", String(mn)) || continue
        grp = f[mn]
        @assert exists(grp, "synamp_lag") || exists(grp, "synamp")
        @assert !exists(grp, "select_threshold")
    end
end
println("OK")
```

- [ ] **Step 2: 跑验证确认失败**

Run: `julia --project=. tmp/verify_input.jl`
Expected: FAIL (`preprocess` group 不存在)

- [ ] **Step 3: 实现 input.jl 重构**

在 `scripts/input.jl` 中,`preprocess_module` 调用前插入 **Layer 0 共享预处理**:

```julia
# === Layer 0: shared full-waveform preprocessing ===
# Per channel: demean -> detrend -> taper -> bandpass (per band)
# Persist to /preprocess/{ch_id}/{band}/obs and /gf_preprocessed/{depth}/{ch_id}/{band}/gf
prepro_obs = Dict{String, Dict{Int, Vector{Float64}}}()      # ch_id -> band -> wf
prepro_gf = Dict{Float64, Dict{String, Dict{Int, Matrix{Float64}}}}()  # depth -> ch_id -> band -> gf
for (ch_id, wf_raw) in channel_data
    prepro_obs[ch_id] = Dict{Int, Vector{Float64}}()
    for (bi, (lo, hi)) in enumerate(band_edges)
        prepro_obs[ch_id][bi] = Signal.preprocess_waveform!(
            copy(wf_raw), dt_stations[ch_id], lo, hi)
    end
end
for depth_val in depths
    prepro_gf[depth_val] = Dict{String, Dict{Int, Matrix{Float64}}}()
    for (ch_id, gf_raw) in gf_data[depth_val]
        prepro_gf[depth_val][ch_id] = Dict{Int, Matrix{Float64}}()
        for (bi, (lo, hi)) in enumerate(band_edges)
            g = copy(gf_raw)
            for c in 1:size(g, 2)
                g[:, c] = Signal.preprocess_waveform!(g[:, c], dt_stations[ch_id], lo, hi)
            end
            prepro_gf[depth_val][ch_id][bi] = g
        end
    end
end
```

`preprocess_module` 改:传 `prepro_obs`/`prepro_gf`(已预处理)替代 `channel_data`/`gf_data`。各算子 `process` 接已预处理波形。

参数写入段(L336-340)改:

```julia
if isdefined(mod, :max_lag_periods)
    cfg_entry["max_lag_periods"] = Float64(mod.max_lag_periods())
end
if isdefined(mod, :source_duration)
    cfg_entry["source_duration"] = Float64(mod.source_duration())
end
if isdefined(mod, :pre_P)
    cfg_entry["pre_P"] = Float64(mod.pre_P())
    cfg_entry["post_P"] = Float64(mod.post_P())
    cfg_entry["pre_S"] = Float64(mod.pre_S())
    cfg_entry["post_S"] = Float64(mod.post_S())
end
# select_threshold/deselect_threshold: deleted, no write
```

持久化 Layer 0 中间结果(write_database 段加):

```julia
# /preprocess/{ch_id}/{band}/obs
pp_gr = create_group(f, "preprocess")
for (ch_id, bands) in prepro_obs
    cg = create_group(pp_gr, ch_id)
    for (bi, wf) in bands
        cg[string(bi)] = wf
    end
end
# /gf_preprocessed/{depth}/{ch_id}/{band}/gf
gpg = create_group(f, "gf_preprocessed")
for (depth_val, chs) in prepro_gf
    dg = create_group(gpg, string(depth_val))
    for (ch_id, bands) in chs
        cg = create_group(dg, ch_id)
        for (bi, gf) in bands
            cg[string(bi)] = gf
        end
    end
end
```

(注:`band_edges`、`dt_stations` 需在 input.jl 前段从 `/paraspace/frequency` + station 解析构建。`preprocess_module` 的 ctx NamedTuple 加 `prepro_obs`、`prepro_gf` 字段替 `channel_data`、`gf_data`。)

- [ ] **Step 4: 跑 e2e 验证**

Run: `julia --project=. scripts/input.jl --data-dir examples/synthetic && julia --project=. tmp/verify_input.jl`
Expected: input.jl 跑通,verify OK

- [ ] **Step 5: format check**

Run: `bash format.sh --check`

______________________________________________________________________

### Task 6: config + doc 更新

**Files:**

- Modify: `config_sample.jl`, `examples/synthetic/config.jl`

- Modify: `doc/schema.md`, `doc/misfit-decomposition.md`, `doc/modules/waveform_proc.md`

- Modify: `shared/config/AGENTS.md`, `shared/misfit/AGENTS.md`(若有)

- [ ] **Step 1: 更新 config 文件**

`config_sample.jl` + `examples/synthetic/config.jl`:

- `Config.XcorrP.maxlag_factor() = 0.5` -> `Config.XcorrP.max_lag_periods() = 3.0`(及 XcorrS/AbsShiftP/AbsShiftS)

- 删 `Config.XcorrP.select_threshold()`/`deselect_threshold()` 行

- `Config.PolarityP.trim() = [0.0, 2.0]` -> `Config.PolarityP.source_duration() = 2.0`

- channel 标签:若有 `channel = "H"`/`"V"` 改 `"N"`/`"E"`/`"Z"`(synthetic 当前无 channel 过滤实例,可跳过或加示例)

- [ ] **Step 2: 更新 doc/schema.md**

- L76-77:Xcorr 参数列表删 select/deselect,maxlag_factor->max_lag_periods,加 source_duration(Polarity)

- L269:删 `selected` 字段行

- 新增 `/preprocess`、`/gf_preprocessed` schema 段(Layer 0 中间结果)

- 新增 Xcorr per-lag `synamp_lag`/`dot_obs_gf_lag` schema

- 新增 PSR `amp_P`/`amp_S`/`obs_psr` schema

- [ ] **Step 3: 更新 doc/misfit-decomposition.md**

- §9:`AbsShiftP/SH/SV` -> `AbsShiftP/Z/N/E`(或保留 P + Z/N/E),channel 标签 "H"/"V" -> "Z"/"N"/"E",删旋转相关

- L154-164,299:stub 删 select/deselect,maxlag_factor->max_lag_periods

- L251:channel 注释 "H"/"V" -> "Z"/"N"/"E"

- [ ] **Step 4: 更新 doc/modules/waveform_proc.md**

各模块预处理表:Xcorr 改 per-lag + 共享 Layer 0;Polarity 改 source_duration + 基础清理;PSR 改 freq-dependent + 接入;新增 Layer 0 共享预处理段。

- [ ] **Step 5: 更新 shared/config/AGENTS.md**

L41:Xcorr 模板参数表删 select/deselect,maxlag_factor->max_lag_periods;Polarity 加 source_duration。
L91-100:示例更新。

- [ ] **Step 6: format check**

Run: `bash format.sh --check`

______________________________________________________________________

### Task 7: 全量验证

**Files:** 无(验证 only)

- [ ] **Step 1: 单测全跑**

Run: `for p in signal misfit io config mt aggregate; do julia --project=shared/$p shared/$p/test/runtests.jl || break; done`
Expected: 全 PASS

- [ ] **Step 2: e2e synthetic**

Run: `julia --project=. scripts/input.jl --data-dir examples/synthetic`
Expected: 跑通,database.h5 含新 schema

- [ ] **Step 3: database.h5 结构核查**

Run: `julia --project=. tmp/verify_input.jl`
Expected: OK

- [ ] **Step 4: format 全检**

Run: `bash format.sh --check`
Expected: 无改动

- [ ] **Step 5: 文档交叉检查**

人工核查:`doc/schema.md`、`doc/misfit-decomposition.md`、`doc/modules/waveform_proc.md`、per-module `AGENTS.md` 与代码一致。

______________________________________________________________________

## Self-Review

**1. Spec coverage:**

- #1 Xcorr 窗/周期数/非对称/max_lag_periods/删 select/deselect -> Task 1(trim_time_window!), Task 2, Task 5(参数写入), Task 6(doc)
- #2 互相关 per-lag reductions + 持久化裁窗 -> Task 2(preprocess), Task 5(持久化)
- #3 Polarity source_duration + 基础清理 -> Task 3, Task 5, Task 6
- #4 PSR freq-dep + 周期数窗 + 接入 -> Task 4, Task 5, Task 6
- #5 三分量 N/E/Z 不旋转 -> Task 6(doc §9 + config 标签)
- #6 共享 pipeline + Layer 0 持久化中间结果 -> Task 1(preprocess_waveform!), Task 5(Layer 0 + 持久化)

**2. Placeholder scan:** 无 TBD/TODO。Task 5 的 `band_edges`/`dt_stations` 构建标注了来源(/paraspace/frequency + station),非占位。

**3. Type consistency:** `preprocess_waveform!` 签名一致(Task 1 定义,Task 5 调用)。`max_lag_periods`/`source_duration`/`pre_P` 等命名跨 task 一致。`synamp_lag`/`dot_obs_gf_lag` 命名一致(Task 2 定义,Task 5 持久化)。

**4. 风险:** Task 5 `band_edges`/`dt_stations` 构建需核查 input.jl 前段是否已有(/paraspace/frequency 解析);若无需补建。Task 4 PSR `process` 的 P/S phase 配对逻辑需核查 phases.txt/picks 结构。
