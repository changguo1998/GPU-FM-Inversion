# Stage: `scripts/input.jl` — Data Ingestion & Initialization

## Role

Runs once at the start of the pipeline (before the main loop). Reads `config.jl`, locates external data (waveforms, station metadata, phase picks, Green's functions) via `Config.load_*()` interface, preprocesses all data into `database.h5` using `shared/signal/` for filtering/trimming and `shared/io/` for HDF5 I/O, and writes the initial strategy into `status_0.h5`.

## 执行流程

```
1. 引导配置
   └─ include(config.jl) → Config.use_misfit!() 注册插件,
                           Config.misfit_modules() (auto), freq_bands(), depths()
   └─ Config.phase_fields()/polarity_fields() 定义震相→字段映射
   └─ 插件可声明 phase_type="P"/"S", 通过 Config.phase_type() 查询

   2. 读外部数据
   └─ Config.load_event() + load_phase_picks() + load_stations()
   └─ 生成 phase_list: 从震相文件提取, 缺失的震相/台站直接跳过
   └─ phase_types 从 phase_list 动态提取, 不硬编码 ("P", "S")

3. 构建 /station 表 (6 行物理台站)
   └─ 按 station.id 去重, 保持原始顺序
   └─ 扁平数组: id, network, lat/lon, dt, distance, azimuth, P/S_time, P_polarity
   └─ channel 信息由 /{ModuleName}/channel_id 的 .Z/.N/.E 后缀隐式携带, 不单独存储

4. 加载原始波形 → /channel
   └─ Config.load_waveform(pid) 逐相位, 去重后存 channel_data{ch_id → Float64[N]}

5. 加载格林函数 → /gf
   └─ Config.load_gf() 逐深度×通道, 存 gf_data{depth → ch_id → Float64[N×6]}

6. 预处理波形 (Layer 0 共享预处理 + 算子 process)
   ├─ Layer 0: 对每个 freq-dependent 模块的频带, 对 obs 逐道 + GF 逐分量
   │    Signal.preprocess_waveform! (demean/detrend/taper + butterworth bandpass)
   ├─ 算子 process(): XcorrP/S 输出 obs/obs_norm2 + per-lag
   │    synamp_lag[depth][band] + dot_obs_gf_lag[band]
   └─ 各深度独立预处理 GF
       (Polarity/Psr 分支 deferred — XCorr-only 模式: basic-clean GF /
        极性窗口 / obs_psr 路径均已移除)


7. 组装字典
   ├─ paraspace: strike/dip/rake 展开 (0:5:355 / 0:5:90 / -90:5:90), depth, frequency(unique排序)
   ├─ 频带选择由各模块 band_low()/band_high() (/config 内索引) 完成
   ├─ event_dict: 坐标/震级/发震时刻
   └─ db_config: misfit_modules, n_bands,
                  {ModuleName}/{params}  ← 按模块名分组的配置, 不含任何索引

8. 写入 database.h5
   └─ 组装 module_data::Dict{String, IO.ModuleData}
   └─ 遍历 misfit_modules (Level 1), result_to_moduledata(r) 映射各 reductions
   └─ IO.write_database(db_path, db_config, event, station, channel, gf,
                         module_data; paraspace = paraspace)
   └─ 每个模块的 channel_id + station_idx 由 ModuleData 携带, 自动写入

9. 写入 status_0.h5
   └─ IO.Strategy(Grid.default_grid() 全空间 5° SDR 网格,
                   depth_indices=[1..n_depths], freq_indices=[1..n_bands], iteration=0)
   └─ IO.write_strategy(status0_path, strategy)
```

### 关键设计

- **三层分离**: `/paraspace` 存值, `/config` 存参数(无索引), `/strategy` 存索引
- **频率**: `Config.freq_bands()` 提供 (low,high) 对 → unique 排序 → `/paraspace/frequency`；模块经 `band_low`/`band_high` (`/config`) 选带，`freq_indices` (`/strategy`) 定义迭代搜索范围
- **各深度 GF 独立预处理**: 不再复用第一个深度
- **config 无索引**: `band_low`/`band_high`/`freq_indices` 为整数索引，分别存于 `/config` 与 `/strategy`

当前已完成：数据接入 (input.jl) + Layer 0 共享预处理 + Misfit 算子 (Xcorr 活跃；Polarity/Psr **deferred**) + aggregate 两级聚合；`preprocess.jl`/`assess.jl`/`output.jl`/`driver.sh` 全部已实现，管道单迭代闭环贯通（**XCorr-only 模式**）。输出 `database.h5` 和 `status_0.h5` 作为后续阶段的接口契约。

## Inputs

| Source | Description |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config.jl` | Bootstrap configuration: frequency bands, depth range, initial grid params, module settings, paths to external data (waveforms, station metadata, Green's functions) |

## Outputs

| Source | Description |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `database.h5` | All preprocessed data: GF at all depths, filtered waveform variants, per-module preprocessing, algorithm config (`/config`, **no indices**), expanded float arrays (`/paraspace`) |
| `status_0.h5` | Initial strategy (`/strategy`) — integer indices referencing `/paraspace`. No trials yet. |

## 三层分离设计

| 位置 | 存什么 | 示例 |
|--------------|-----------------------------------------|---------------------------------------------------------------------------|
| `/paraspace` | 展开的浮点值 (`Float64[N]`) | `strike[71]`, `dip[19]`, `rake[37]`, `depth[3]`, `frequency[2]` |
| `/config` | 算法参数和元数据，**无索引无浮点参数值** | `misfit_modules`, `{ModuleName}/trim`, `max_lag_periods`, `band_low/high` |
| `/strategy` | 整数索引 (`Int32[N]`) 指向 `/paraspace` | `depth_indices[3]`, `freq_indices[2]`, `iteration` |

关于频率：`Config.freq_bands()` 返回 `[(low, high), ...]`，input.jl 提取所有唯一边界、排序后写入 `/paraspace/frequency`。Freq-dependent 模块（XCorr/Psr）通过 `band_low()`/`band_high()` 指向该数组；`/strategy` 保存 `freq_indices`（1..N_bands）作为迭代搜索范围。

## Responsibilities

1. **Preprocess raw data**: filter waveforms to frequency bands, trim time windows, extract XCorr preprocessing output (Polarity 分支已 deferred), store in `database.h5`
1. **Load Green's functions**: read external GF files, store by phase × depth in `database.h5`
1. **Write algorithm config**: load `config.jl`, write `db_config` (module list, module params) into `database.h5` — **no indices, no float parameter values**
1. **Write expanded parameter space**: compute grid axis expansions (strike/dip/rake) via `Grid.expand_axis()`, build `frequency` from unique band edges, store all as `/paraspace` in `database.h5`
1. **Write initial strategy**: build `IO.Strategy(depth_indices, freq_indices, iteration=0)` → `/strategy` in `status_0.h5`
1. **Write phase metadata**: write `channel_id` + `station_idx` into each `/{ModuleName}` group in `database.h5` (carried by `ModuleData`)
1. **Create file skeleton**: `status_0.h5` is created with `/strategy` populated.
1. **Per-depth GF preprocessing**: each trial depth independently filters and windows its own Green's functions during XCorr preprocessing (Polarity 已 deferred; 此前所有深度复用第一个深度的 GF).

## Script Style

Flat, straight-line script — no `main()` wrapper. Runs top-down when `include`d or executed.

Tooling functions (time parsing, distance/azimuth computation, phase ID extraction) live in `shared/io/` (module `IO`) and are called as `IO.parse_time_iso`, `IO.haversine_distance`, etc.

- Julia (`HDF5.jl`, `DSP.jl` via `shared/signal/`, `Dates.jl`). Psr 算子 **deferred**（XCorr-only 模式）——其 `process()` 计算 `obs_psr`/`amp_P`/`amp_S` 的路径已移除，恢复时按 git HEAD 0a9ad69 重新接线。
- Butterworth bandpass filter (DSP.jl, zero-phase forward-backward)
- Time-window trimming
- Green's function loader

## What It Does NOT Do

- Does NOT generate trials (future `preprocess.jl`)
- Does NOT compute misfits (future forward stage)
- Does NOT apply weights or make strategy decisions (future `assess.jl`)
- Does NOT run more than once per pipeline invocation
