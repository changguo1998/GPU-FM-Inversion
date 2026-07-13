# Stage: `scripts/input.jl` — Data Ingestion & Initialization

## Role

Runs once at the start of the pipeline (before the main loop). Reads `config.jl`, locates external data (waveforms, station metadata, phase picks, Green's functions) via `Config.load_*()` interface, preprocesses all data into `database.h5` using `shared/signal/` for filtering/trimming and `shared/io/` for HDF5 I/O, and writes the initial strategy into `status_0.h5`.

## 执行流程

```
1. 引导配置
   └─ include(config.jl) → Config.use_misfit!() 注册插件,
                           Config.misfit_modules() (auto), freq_bands(), depths()

2. 读外部数据
   └─ Config.load_event() + load_phase_picks() + load_stations()
   └─ 验证: 每个 station 必须有对应的 phase pick
   └─ 生成 phase_list: 每站 P/S 两条, 含 phase_id + phase_type + station_idx

3. 构建 /station 表
   └─ 扁平数组: id, network, channel, lat/lon, dt, distance, azimuth, P/S_time, P_polarity

4. 加载原始波形 → /channel
   └─ Config.load_waveform(pid) 逐相位, 去重后存 channel_data{ch_id → Float64[N]}

5. 加载格林函数 → /gf
   └─ Config.load_gf() 逐深度×通道, 存 gf_data{depth → ch_id → Float64[N×6]}

6. 预处理波形 (逐频带、逐相位类型)
   ├─ 按 misfit_modules 遍历, 每个模块实例独立预处理
   ├─ XCorrP/XcorrS: Config.XcorrP.preprocess() 等
   │          输出 obs[np×nt], gf[np×6×nt], synamp[np×6×6]
   │          各深度独立预处理 GF
   └─ Polarity: Config.Polarity.preprocess() 极性窗口
                 输出 obs[nc×1], gf[nc×6×npolarity_samples]

7. 组装字典
   ├─ paraspace: strike/dip/rake 展开 (0:5:355 / 0:5:90 / -90:5:90), depth, frequency(unique排序)
   ├─ freq_low_idx/high_idx: 每个频带的低/高切在 frequency[] 中的位置
   ├─ event_dict: 坐标/震级/发震时刻
   └─ db_config: misfit_modules, n_bands,
                  {ModuleName}/{params}  ← 按模块名分组的配置, 不含任何索引

8. 写入 database.h5
   └─ 组装 module_data::Dict{String, IO.ModuleData}
   └─ IO.write_database(db_path, db_config, event, station, channel, gf,
                         module_data; paraspace = paraspace)
   └─ 每个模块的 channel_id + station_idx 由 ModuleData 携带, 自动写入

9. 写入 status_0.h5
   └─ IO.Strategy(depth_indices=[1..n_depths],
                   freq_low_idx, freq_high_idx, iteration=0)
   └─ IO.write_strategy(status0_path, strategy)
```

### 关键设计

- **三层分离**: `/paraspace` 存值, `/config` 存参数(无索引), `/strategy` 存索引
- **频率**: `Config.freq_bands()` 提供 (low,high) 对 → unique 排序 → `/paraspace/frequency` → 计算 `freq_low_idx`/`freq_high_idx` 索引对
- **各深度 GF 独立预处理**: 不再复用第一个深度
- **config 无索引**: `freq_low_idx`/`freq_high_idx` 只在 `/strategy` 中

当前为从头开发的第一阶段。输出 `database.h5` 和 `status_0.h5` 作为后续阶段的接口契约。

## Inputs

| Source | Description |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config.jl` | Bootstrap configuration: frequency bands, depth range, initial grid params, module settings, paths to external data (waveforms, station metadata, Green's functions) |

## Outputs

| Source | Description |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `database.h5` | All preprocessed data: GF at all depths, filtered waveform variants, per-module preprocessing, algorithm config (`/config`, **no indices**), expanded float arrays (`/paraspace`) |
|| `status_0.h5` | Initial strategy (`/strategy`) — integer indices referencing `/paraspace`. No trials yet. |

## 三层分离设计

| 位置 | 存什么 | 示例 |
|--------------|-----------------------------------------|------------------------------------------------------------------------|
| `/paraspace` | 展开的浮点值 (`Float64[N]`) | `strike[71]`, `dip[19]`, `rake[37]`, `depth[3]`, `frequency[2]` |
| `/config` | 算法参数和元数据，**无索引无浮点参数值** | `misfit_modules`, `n_bands`, `xcorr/maxlag_factor`, `polarity/trim` |
| `/strategy` | 整数索引 (`Int32[N]`) 指向 `/paraspace` | `depth_indices[3]`, `freq_low_idx[1]`, `freq_high_idx[1]`, `iteration` |

关于频率：`Config.freq_bands()` 返回 `[(low, high), ...]`，input.jl 从中提取所有唯一值、排序后写入 `/paraspace/frequency`，然后计算每个频带的 `freq_low_idx[i]` / `freq_high_idx[i]`（即 low/high 在 frequency 数组中的位置）存入 `/strategy`。没有 `freq_indices` — 所有频带由 N_bands 对索引隐式定义。

## Responsibilities

1. **Preprocess raw data**: filter waveforms to frequency bands, trim time windows, extract XCorr and Polarity preprocessing output (GF preprocessed independently per depth), store in `database.h5`
1. **Load Green's functions**: read external GF files, store by phase × depth in `database.h5`
1. **Write algorithm config**: load `config.jl`, write `db_config` (module list, module params) into `database.h5` — **no indices, no float parameter values**
1. **Write expanded parameter space**: compute grid axis expansions (strike/dip/rake) via `Grid.expand_axis()`, build `frequency` from unique band edges, store all as `/paraspace` in `database.h5`
1. **Write initial strategy**: build `IO.Strategy(depth_indices, freq_low_idx, freq_high_idx, iteration=0)` → `/strategy` in `status_0.h5`
1. **Write phase metadata**: write `channel_id` + `station_idx` into each `/{ModuleName}` group in `database.h5` (carried by `ModuleData`)
1. **Create file skeleton**: `status_0.h5` is created with `/strategy` populated.
1. **Per-depth GF preprocessing**: each trial depth independently filters and windows its own Green's functions during XCorr/Polarity preprocessing (previously all depths reused the first depth's GF).

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down when `include`d or executed.

Tooling functions (time parsing, distance/azimuth computation, phase ID extraction) live in `shared/io/` (module `IO`) and are called as `IO.parse_time_iso`, `IO.haversine_distance`, etc.

- Julia (`HDF5.jl`, `DSP.jl` via `shared/signal/`, `Dates.jl`). PSR preprocessing not called by current input.jl (no PSR data stored in database.h5).
- Butterworth bandpass filter (DSP.jl, zero-phase forward-backward)
- Time-window trimming
- Green's function loader

## What It Does NOT Do

- Does NOT generate trials (future `preprocess.jl`)
- Does NOT compute misfits (future forward stage)
- Does NOT apply weights or make strategy decisions (future `assess.jl`)
- Does NOT run more than once per pipeline invocation
