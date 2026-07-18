# 算子预处理设计 (2026-07-17)

## 1. 背景

延续 2026-07-17 session 的算子预处理讨论(handoff: `tmp/handoff-refactor-fm-preprocessing.md`)。实际算子 3 类:Xcorr、Polarity、PSR(deferred)。AbsShift 复用 Xcorr kernel(取 BEST_LAG),RelShift composed。六个设计点全定稿。

## 2. 架构分层

预处理分两层:

- **Layer 0 共享预处理**(input.jl 内,非算子):对 obs + GF 完整波形做 去均值 -> 去趋势 -> 尖灭(taper) -> bandpass(各频带)。独立于算子,所有 freq-dependent 算子共用。中间结果(各频带完整滤波波形)持久化到 database.h5 供调试。
- **Layer 1 算子 reductions**:各算子消费 Layer 0 输出(已预处理波形),计算各自 reductions,持久化。算子不再内联滤波。

算子 `process` 签名改为接收已预处理波形,只算 reductions。

## 3. Xcorr(#1 + #2)

### 3.1 窗与 max_lag(#1)

- `trim()` 语义:秒 -> 周期数(无量纲)。窗长 `pre_sec = trim[1]/band_high`,`post_sec = trim[2]/band_high`,前后独立(非对称)。
- 裁窗 `[arrival - pre_sec/dt, arrival + post_sec/dt]`。
- 删 `wf_filter = max(pre,post)*high_cut` 与 `trim_time_window!` 对称半窗逻辑;`trim_time_window!` 改签名接 `(pre_periods, post_periods, band_high)`,非对称裁窗。
- `maxlag_factor()` -> `max_lag_periods()`:周期数,`max_lag_sec = periods/band_high`,`max_lag_samples = min(round(max_lag_sec/dt), (nt-1)÷2)`。
- 删 `select_threshold()`/`deselect_threshold()` stub + export + input.jl 写入。
- 删 schema `selected` 字段。取消硬阈值,所有 phase 等权参与 aggregate,无 selected 标记,无 phase 权重。

### 3.2 互相关机制(#2)

- obs 固定 trim 窗,syn(GF 组合)从完整滤波波形取,lag 滑动 `[-max_lag, +max_lag]`,滑出 obs 窗部分用完整波形相邻部分补(非零 pad)。
- `CC(lag) = obs[窗] · syn_full[窗-lag]`,lag 不越完整波形 -> 无 pad,obs 窗等长 -> 无截断。
- #2 原 `nt_xc = minimum` 截断问题化解。

### 3.3 per-lag reductions(预计算)

kernel 公式(per trial M,per lag):

```
CC(lag)        = M · dot_obs_gf(lag)
syn_norm2(lag) = M' · synamp(lag) · M
CC_norm(lag)   = CC(lag) / sqrt(obs_norm2 · syn_norm2(lag))
CC_MAX         = max_lag CC_norm(lag)
BEST_LAG       = argmax_lag CC_norm(lag)
```

预计算 reductions(per lag, per depth, per band):

- `synamp(lag) = GF[窗-lag]' · GF[窗-lag]` (6×6)
- `dot_obs_gf(lag) = obs[窗] · GF[窗-lag]` (6 向量)
- `obs_norm2 = ‖obs[窗]‖²` (标量)

持久化:

- `obs[N,nt_xc]`、`gf[N,6,nt_xc]` 裁窗波形(调试)
- `obs_norm2[N]`
- `synamp(lag)[N,6,6,L]`、`dot_obs_gf(lag)[N,6,L]`(`L = 2*max_lag_samples+1`)

GF 完整滤波波形:Layer 0 持久化(调试 + synamp/dot_obs_gf 计算源)。

## 4. Polarity(#3)

- `is_freq_dependent() = false`。
- `trim()` 分离:改用 `source_duration()` 秒(因无 band_high,不能用周期数)。config:`Config.PolarityP.source_duration() = 2.0`。
- GF 预处理:基础清理(去均值/去趋势/taper,无 bandpass),用 Layer 0 的 `preprocess_waveform!` 但 bandpass 关闭。
- `obs_pol` 不预处理(人工拾取 ±1/NaN)。
- 初动符号从清理后 GF 取。
- 持久化:`obs_pol[N]`、`gf_pol[N,6,n_pol]`(band 1, per depth)。

## 5. PSR(#4)

- `is_freq_dependent = true`。新建 `shared/misfit/src/Psr.jl` + 注册 `use_misfit!` + input.jl 支持。
- 预处理:Layer 0 共享(完整波形 去均值/去趋势/taper/bandpass 各频带)。
- P/S 分别裁窗算 rms,窗 `pre_P/post_P/pre_S/post_S` 周期数,窗长 = `periods/band_high` 随频带变。
- reductions:`amp_P = GF_P'·GF_P` (6×6),`amp_S = GF_S'·GF_S`,`obs_psr = log10(rms_P_obs/rms_S_obs)`,per band。
- kernel:`syn_psr = 0.5·log10(M'·amp_P·M / M'·amp_S·M)`,`misfit = obs_psr - syn_psr`。

## 6. 三分量(#5)

- 不旋转,N/E/Z 三道独立互相关。
- `channel` 过滤机制保留(`use_misfit!` channel 参数),标签 `"Z"`/`"N"`/`"E"`(替代 doc §9 `"H"`/`"V"`)。
- 实例 `{Operator}{Phase}_{Z/N/E}`,具体组合由 config 决定。
- doc §9 修订:SH/SV -> Z/N/E,删旋转相关描述。
- RelShift = StdDev 跨 channel AbsShift(同 station)。

## 7. 共享 pipeline(#6)

- Signal 层:`preprocess_waveform!(wf, dt, low_cut, high_cut; demean, detrend, taper, order)` 组合去均值/去趋势/taper/bandpass。
- 原子函数保留:`bandpass_filter!`、`demean!`、`detrend!`、`taper!`、`trim_time_window!`、`trim_to_polarity_window!`。
- 删旧 `Signal.preprocess_xcorr!`/`preprocess_polarity!`/`preprocess_psr!`(被共享 + 算子 reductions 替代)。
- Polarity 用 `preprocess_waveform!` 但 bandpass 关闭(只 demean/detrend/taper)。

## 8. 持久化 schema 变化(database.h5)

新增:

- `/preprocess/{channel_id}/{band}/obs`:各频带完整滤波 obs(调试)
- `/gf_preprocessed/{depth}/{channel_id}/{band}/gf`:各频带完整滤波 GF(调试 + reductions 源)
- Xcorr per-lag:`synamp(lag)`、`dot_obs_gf(lag)`
- PSR:`amp_P`、`amp_S`、`obs_psr`(per band)
- Polarity:`source_duration` 参数

删除:

- `select_threshold`、`deselect_threshold`(/config)
- `selected` 字段(status/output)

修改:

- `maxlag_factor` -> `max_lag_periods`(/config)
- `trim` 语义 秒 -> 周期数(/config)
- Xcorr reductions:固定窗 -> per-lag

保留:

- `obs`/`gf` 裁窗(调试)

## 9. 影响清单

| 文件 | 改动 |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `shared/misfit/src/Xcorr.jl` | trim 周期数;删 select/deselect;maxlag->max_lag_periods;preprocess 改 reductions(per-lag);process 接已预处理波形 |
| `shared/misfit/src/Polarity.jl` | trim->source_duration;preprocess 接已预处理波形(基础清理) |
| `shared/misfit/src/Psr.jl` | 新建 |
| `shared/misfit/src/Misfit.jl` | 注册 Psr 模块 |
| `shared/signal/src/Signal.jl` | 新增 `preprocess_waveform!`、`demean!`、`detrend!`、`taper!`;删 `preprocess_xcorr!/polarity!/psr!`;`trim_time_window!` 改非对称 |
| `scripts/input.jl` | Layer 0 共享预处理 + 持久化中间结果;Layer 1 算子 reductions;删 select/deselect 写入;maxlag->max_lag_periods |
| `shared/config/*` | `use_misfit!` channel 标签 Z/N/E;Polarity `source_duration`;Xcorr `max_lag_periods` |
| `doc/schema.md` | 删 selected;maxlag->max_lag_periods;trim 语义;新增 per-lag/PSR/预处理中间结果 schema |
| `doc/misfit-decomposition.md` | §9 SH/SV->Z/N/E;删旋转;select/deselect 删 |
| `doc/modules/waveform_proc.md` | 各模块预处理表更新 |
| `config_sample.jl`、`examples/synthetic/config.jl` | maxlag->max_lag_periods;删 select/deselect;Polarity source_duration;channel Z/N/E |

## 10. 实现路径

1. Signal 层:新增 `preprocess_waveform!` + `demean!`/`detrend!`/`taper!`;改 `trim_time_window!` 非对称;删旧 `preprocess_*`。
1. Xcorr.jl:trim 周期数;`max_lag_periods`;删 select/deselect;preprocess 改 per-lag reductions;process 接已预处理波形。
1. Polarity.jl:`source_duration`;preprocess 接已预处理波形。
1. Psr.jl:新建 + 注册。
1. input.jl:Layer 0 共享预处理 + 持久化中间结果;Layer 1 算子 reductions;参数写入更新。
1. config + doc 更新。
1. 验证:examples/synthetic e2e + database.h5 diff。

## 11. 待定/风险

- per-lag reductions 数据量:`L = 2*max_lag_samples+1`,如 periods=3/band_high=0.5Hz/dt=0.1s -> L≈121。`synamp(lag)` 增 ~L 倍。需评估 database.h5 大小。
- 预处理中间结果持久化:各频带完整 obs+GF,数据量大。调试用,可选 band/station 子集。
- backazimuth/入射角:三分量不旋转,无需。但若未来加旋转,需台站-震源方位角数据。
- forward kernel(C++):未实现 per-lag reductions 消费,需后续改造。
