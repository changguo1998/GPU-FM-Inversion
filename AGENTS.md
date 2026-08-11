# AGENTS.md — Focal Mechanism Inversion (从头开发)

## Project identity

震源机制反演管道。Julia 数据接入 + 预处理（Layer 0 共享预处理 + 算子 reductions），HDF5 数据交换。已完成数据接入、Misfit 算子、aggregate 两级聚合与**全管道贯通**（单迭代闭环）；多迭代细化（assess 权重聚合/网格细化）待开发。

## Project layout

```
scripts/        Flat stage scripts (input/preprocess/assess/output — 全部已实现)
shared/         Julia packages by function (io, mt, grid, signal, aggregate, misfit, config, stage_log)
config_sample.jl   Template pipeline configuration
```

## Per-module reference

| Module | AGENTS.md | Role |
|-----------|------------------------------|----------------------------------------------|
| IO | `shared/io/AGENTS.md` | HDF5 I/O, type structs, geophysics utilities |
| MT | `shared/mt/AGENTS.md` | SDR ↔ MT conversion |
| Grid | `shared/grid/AGENTS.md` | Trial generation + grid refinement |
| Signal | `shared/signal/AGENTS.md` | Waveform preprocessing |
| Aggregate | `shared/aggregate/AGENTS.md` | Misfit aggregation, uncertainty |
| Config | `shared/config/AGENTS.md` | Pipeline configuration interface |
| StageLog | `shared/stage_log/AGENTS.md` | Per-stage logging |

## 开发路线 (Development approach)

围绕一个事例数据（`examples/synthetic`），先以 **XCorr 为唯一目标函数** 打通并打磨全流程
（input → preprocess → forward → assess → output），最后再扩展其他算子（Polarity/Psr）。

- 当前所有开发决策以 `examples/synthetic` 为准：改动必须保持 XCorr 全流程在该事例上端到端可跑，best 结果可复现。
  - **验收基线（修复 mt 布局 bug 后，2026-08-10）**：72 网格全链 best = **(30,65,-90) @ 10 km, misfit ≈ 0.1593**（XCorrS-only，9 通道）。合成数据真实源为 (30,60,90,10 km)（`tests/synthetic_data.jl` DEFAULT\_\*）；rake ±90 为 |cc_max| 无极性下的互补节点，misfit 相同；dip 60 vs 65 因数据噪声/台站几何仅差 ~0.009。
  - **历史警告**：2026-08-10 前所有 e2e 的 best #29522 (65,90,85, 0.15518) 均由 XCorr kernel 的 **MT 布局 bug**（kernel 列主读 `mt[trial + c*N]`、main 行主写 `mt[trial*6+c]`）产生，已作废。该 bug 与 trial 规模耦合（n_sub=1 时行列主等价掩盖），在 strike 72 网格（n_sub=50616）暴露为"misfit 错乱"；修复后 71/72 网格结果完全一致。
- XCorr 一族（XcorrP/S、AbsShiftP/S、RelShift）为活跃目标函数/派生诊断；Polarity/Psr 属"算子扩展"阶段，恢复时按 `git HEAD 367dfd1` 前的注册与预处理接线为准。

## 当前阶段

已完成：`input.jl` 数据接入与初始化（Layer 0 共享预处理 + 算子 reductions）、Misfit 算子 (Xcorr 活跃；Polarity/Psr 已实现但 **deferred**，XCorr-only 模式)、aggregate 两级聚合 (extractors/composers/StdDev)、`assess.jl` (extract+compose+收敛决策)、`preprocess.jl` 试次生成、`output.jl` 输出编译、`driver.sh` 全管道贯通（单迭代收敛，XCorr-only）、**XCorr kernel MT 布局 bug 修复（2026-08-10）**、trials 全参数索引化（strike/dip/rake/depth/freq 均以 `/paraspace` 索引表示）。待开发：assess 权重聚合/网格细化（多迭代闭环）。

```
scripts/input.jl  (一次) → database.h5 + status_0.h5
```

后续阶段（preprocess → forward → assess → output）持续推进，接口契约由 `database.h5` 和 `status_N.h5` schema 定义（见 `doc/schema.md`）。

## HDF5 files

| File | Lifetime | Contents |
|-----------------|---------------|----------------------------------------------------------------------------------------------------------|
| `database.h5` | Static | Greens at all depths, all freq-band variants, per-module preprocessed data, **paraspace**, config, index |
| `status_{N}.h5` | Per-iteration | Strategy, trials, misfits for iteration N |
| `output.h5` | Final | Best-fit parameters, uncertainties, per-phase/station breakdown |

### `/paraspace` (new in database.h5)

Stores expanded float arrays for parameter-space dimensions:
strike/dip/rake (from grid expansion), depth, frequency (discrete values, `Float64[N_freq]`).
Per-module band selection uses `band_low`/`band_high` in `/config/{ModuleName}/` pointing into
`/paraspace/frequency`. Integer indices live in `/config` (module params) and `/strategy` (search dimensions).
See `doc/schema.md` for details.

## Domain concepts

- **Moment tensor**: 6 components in NED: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
- **Source params**: strike \[0,360), dip [0,90], rake [-90,90] (degrees)
- **Green's functions**: 6-component waveforms per station, pre-computed externally
- **Misfit modules**: XCorr active; Polarity/Psr implemented but **deferred** (XCorr-only mode, 2026-08-09). AbsShift = Xcorr BEST_LAG output; RelShift = Aggregate.StdDev composer (registered in sample configs). CAP — cancelled.
- **Trial**: one combination of variable params (SDR, depth, frequency, etc.)
- **Phase** = station + channel + wave type (P/S) — channels subsumed by phases
- **Phase key**: `{network}.{station}.{channel}.{phase_type}` (e.g. `IU.COLA.00.P`)

## Coding assumptions

These conventions apply across the entire project.

### Architecture

1. All frequency-band variants precomputed upfront by `input.jl` — no runtime filtering.
1. Green's functions pre-computed externally — loaded by `input.jl`, never computed by pipeline.
1. Config bootstrapped via `config.jl` (Julia script implementing `Config` module interface) — only `input.jl` reads it. All config written to `database.h5`; subsequent stages read from HDF5.
1. HDF5 schema is the API contract between stages — schema changes require coordinated stage updates.
1. **Flat scripts**: stage scripts execute as straight-line top-level code with no `main()` wrapper. Private helper functions are allowed to flatten deep nesting; they must be self-contained (no leaked state) and stay subordinate to the top-level flow.
1. **Shared packages**: utility code lives in `shared/` Julia packages imported via `using`. Each package has own `Project.toml`.

### Formatting

1. **4-space indent**: all languages. Julia (`.JuliaFormatter.toml` `indent=4`). No tabs.
1. **Space around operators**: spaces on both sides of `=`, `==`, `<`, `>`, `+`, `-`, `*`, etc.
1. **Compact style**: no unnecessary line breaks. Short blocks stay on fewer lines where readable.
1. **Short docstrings**: every exported/public function gets 1-3 line docstring.

### Data conventions

1. All angles in degrees in HDF5 and Julia.
1. All HDF5 datasets `Float64` unless noted. Scalars stored as scalar datasets.

### Workflow

1. **Format before stage or commit**: run `bash format.sh` on all changed files before staging (parallel, covers Julia/Markdown). Use `bash format.sh --check` for dry-run.
1. **Docs follow code**: update or create relevant `doc/` and per-module `AGENTS.md` in same change as code modification, before staging. Stale docs treated as technical debt — no separate "docs PR" later.
1. **Logical commit grouping**: split work into focused, independently-reviewable commits. Each commit message uses conventional prefix: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `build:`.
1. **Parallelize independent work**: fan out independent tasks across available compute budget. Constraint: shared resources (HDF5 file locks, mutable state) serialize where necessary.
1. **Caveman communication by default**: use compressed, article-free, fragment-style responses for all tool-assisted development interactions. Full prose reserved for commit messages, user-facing documentation, and cases where compression creates ambiguity.

### Versioning

1. No CI configured. Tests run manually.
1. No formal semantic versioning — package versions are `0.1.0` placeholders.
