# Module: IO (HDF5 I/O)

**Location**: `shared/io/` (Julia package `IO`)

## Role

提供阶段间共享的类型、HDF5 读写和地理工具。完整数据契约见
`doc/schema.md`；C++ forward 通过 `forward/src/hdf5_io.*` 独立读写同一 schema。

## Types

| Type | Purpose |
|---------------|------------------------------------------------------|
| `EventInfo` | 震源位置、震级和发震时刻 |
| `StationInfo` | 台站、通道、坐标、采样间隔和开始时刻 |
| `PhasePick` | P/S 到时与 P 初动极性 |
| `ModuleData` | 每个 misfit 实例的 obs/GF/reductions 与 phase 元数据 |
| `Strategy` | SDR 网格和 depth/frequency/duration 索引 |
| `TrialSet` | 全参数 trial 索引向量 |

## Main interfaces

- Database: `write_database`, `read_config`, `read_event`, `read_stations`,
  `read_waveform`, `read_greens`.
- Search state: `write_strategy`, `read_strategy`, `write_trials`, `read_trials`.
- Results: `write_misfits`, `read_misfits`, `write_output`.
- Parameter space: `write_paraspace`, `read_paraspace`.
- Utilities: `h5create_group`, `h5exists`, `find_latest_status`,
  `parse_time_iso`, `haversine_distance`, `compute_azimuth`.

## Current data flow

1. `input.jl` 写 `database.h5` 和初始 `status_0.h5:/strategy`。
1. `preprocess.jl` 写当前 status 的 `/trials`。
1. C++ forward 事务性替换 `/intermediates`。
1. `assess.jl` 写 `/misfits` 和收敛决策。
1. `output.jl` 汇总为 `output.h5` 和 `result.toml`。

## Conventions

- HDF5 和 Julia 索引为 1-based；C++ 读取后仅在内部转为 0-based。
- 参数值存于 `/paraspace`，搜索索引存于 `/strategy` 和 `/trials`。
- 阶段写完整 dataset，不使用 append 语义。
