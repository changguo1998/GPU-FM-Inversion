# Stage: `scripts/assess.jl` — Misfit Extraction, Composition, Convergence

## Role

Runs once per iteration (preprocess → forward → **assess**). Reads the raw
intermediates from `status_N.h5:/intermediates/`, transforms them into misfit
matrices (Level 1 extractors), aggregates composed modules (Level 2 composers),
and writes channel/native-entry objective values to `status_N.h5:/misfits/`.
Aggregation is a separate channel → station → trial summation under `/aggregate/`.
Also emits the iteration convergence decision consumed by `driver.sh`.

## Usage

```bash
julia scripts/assess.jl <database.h5> <status_N.h5>
```

When `DATA_DIR` is set in the environment, an empty `.decision.txt` is written
into it (converged). Exit code 0 on success.

## Inputs

| File | Used for |
|---------------|------------------------------------------------------------------------|
| `database.h5` | `/config` (module list + params), `/station`, `/{Module}/station_idx` |
| `status_N.h5` | `/intermediates/{key}/` (read), `/trials` (count), `/misfits/` (write) |

## Processing

1. **Read context**: config modules, stations, trial count.

1. **Level 1 — extract**: for each non-composed module, map its
   `(operator, output)` to an extractor in `Aggregate.EXTRACTORS`:

   | (operator, output) | Extraction |
   |-----------------------|-----------------------------------------------------------------|
   | `(:Xcorr, :cc_max)` | `1 .− cc_max` (normalized-CC misfit) |
   | `(:Xcorr, :best_lag)` | `best_lag · dt` (absolute shift, s) |
   | PSR | `abs2(log(rms(S_obs)/rms(P_obs)) - log(rms(S_syn)/rms(P_syn)))` |
   | normalized polarity | L1 difference of per-trial L2-normalized signed amplitudes |

   Intermediate rows are transposed to `[entries × trials]` (the C++ writes
   C-order `[N_phases × N_trials]`, which HDF5.jl reads reversed).

1. **DSL objectives**：从 `/config/objectives` 恢复表达式树；根据 `bases`
   对齐 channel，直接把 `cc_max`、signed lag、能量、振幅和符号中间量代入表达式。
   基础运算逐元素执行；派生 `energy`/`rms` 按 trial 跨 entry 归约。

1. **Legacy compose**：未使用 DSL 注册的 composed modules 仍按 `bases`
   拓扑执行 `Aggregate.COMPOSERS`，保留兼容接口。

1. **Write `/misfits/`**: one dataset per module, shape `[N_entries × N_trials]`,
   replacing any previous value (idempotent).

1. **Persist indices**: `/misfit_index/{ModuleName}` records channel IDs, station
   indices, and whether rows are phase- or channel-indexed, without reducing the
   trial axis.

1. **Hierarchical sum**: sum phase-indexed values within each physical channel, add
   channel-indexed values, sum channels within each station, add native
   station-indexed values, then sum stations into one value per trial. No mean,
   min-max normalization, or objective weighting is applied. Signed Lag contributes
   its absolute value; raw `/misfits/Lag*` remains signed.

1. **Convergence decision**: current implementation converges on the first
   iteration (writes an empty `.decision.txt`). The multi-iteration strategy
   is pending redesign; see `doc/roadmap.md`.

## Outputs

### `status_N.h5:/misfits/{ModuleName}`

| Dataset | Type | Shape | Description |
|------------------------------------------|---------|--------------------------|----------------------------|
| `XcorrP/S`, `LagP/S`, `Psr`, `PolarityP` | Float64 | `[N_entries × N_trials]` | per-objective value matrix |

### `{DATA_DIR}/.decision.txt`

Empty file = converged (driver exits the loop). Non-empty = continue.

### `status_N.h5:/aggregate`

`channel/{phase_sum,direct_sum,total}` contains the channel-level decomposition;
`station/{channel_sum,direct_sum,total}` contains the station-level decomposition;
`total` contains the final sum per trial.

## Notes

- `read_intermediate` uses the (N_trials, N_entries) heuristic on the read
  shape to normalize the C++ C-order storage into `[entries × trials]`.
- 当前基线注册 XcorrP/S、LagP/S、Psr、PolarityP。Lag 保留方向符号。
- assess does NOT modify `/strategy` or `/trials`.
- Baseline (2026-09-21): hierarchical-sum best trial = (210, 30, 90) @ 10 km,
  σ=0.2 s, misfit ≈ 2.06561e-3；该解与真值 (30, 60, 90) 的 moment
  tensor 相同。
