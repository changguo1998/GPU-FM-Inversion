# AGENTS.md — Focal Mechanism Inversion (从头开发)

## Project identity

震源机制反演管道。Julia 数据接入 + 预处理（Layer 0 共享预处理 + 算子 reductions），HDF5 数据交换，C++ OpenMP/CUDA forward。目标函数 DSL 已支持现有数学原语的通用组合编译；XCorr、signed lag、PSR、归一化极性已完成**单迭代全管道贯通**；Search 已实现预算约束初始采样规划，多迭代区域裁剪待开发。

## Project layout

```
scripts/        Flat stage scripts (input/preprocess/assess/output/report — 全部已实现)
shared/         Julia packages by function (io, mt, search, signal, aggregate, misfit, config, stage_log)
config_sample.jl   Template pipeline configuration
```

## Per-module reference

| Module | AGENTS.md | Role |
|-----------|------------------------------|-----------------------------------------------|
| IO | `shared/io/AGENTS.md` | HDF5 I/O, type structs, geophysics utilities |
| MT | `shared/mt/AGENTS.md` | SDR ↔ MT conversion |
| Search | `shared/search/AGENTS.md` | Parameter-space definition + trial generation |
| Signal | `shared/signal/AGENTS.md` | Waveform preprocessing |
| Aggregate | `shared/aggregate/AGENTS.md` | Misfit aggregation, uncertainty |
| Config | `shared/config/AGENTS.md` | Pipeline configuration interface |
| StageLog | `shared/stage_log/AGENTS.md` | Per-stage logging |

## 开发路线 (Development approach)

围绕事例数据（`examples/synthetic`）维护 XCorr 验收基线，并以目标函数 DSL 组合
`maxCC`/`lagCC`/`energy`/`rms`/`ampScale`/`signScale`。

- 当前所有开发决策以 `examples/synthetic` 为准：改动必须保持 XCorr 全流程在该事例上端到端可跑，best 结果可复现。
  - **当前验收基线（2026-09-21）**：候选 σ = `[0.1, 0.2, 0.3] s`，455,544 trials；分层求和 best = **(210,30,90) @ 10 km, σ=0.2 s, misfit ≈ 2.06561e-3**。该 SDR 是真值 `(30,60,90)` 的辅助节面，moment tensor 完全相同；P/S 全部 best lag = 0。
  - **历史警告**：2026-08-10 前所有 e2e 的 best #29522 (65,90,85, 0.15518) 均由 XCorr kernel 的 **MT 布局 bug**（kernel 列主读 `mt[trial + c*N]`、main 行主写 `mt[trial*6+c]`）产生，已作废。该 bug 与 trial 规模耦合（n_sub=1 时行列主等价掩盖），在 strike 72 网格（n_sub=50616）暴露为"misfit 错乱"；修复后 71/72 网格结果完全一致。
- 基线 config 注册 XcorrP/S、LagP/S、Psr、PolarityP。XCorr/Lag 是基础计算目标；其余表达式编译为 `Expression`，从共享中间量通用求值。assess 将目标值按 channel → station → trial 逐级求和；signed Lag 仅在汇总时取绝对值。

## 当前阶段

已完成：`input.jl` 数据接入与 Layer 0 预处理、目标函数 DSL、XCorr/lag/PSR/归一化极性、trials 全参数索引化、预算约束初始采样规划及 preprocess 接入、Gaussian STF duration 搜索、`assess.jl` 单轮收敛与分层求和、`output.jl`、`report.jl`、`driver.sh` 全管道、OpenMP/CUDA forward（显存复用、分批、preflight、HDF5 事务提交）。待开发：多迭代区域裁剪。

```
scripts/input.jl  (一次) → database.h5 + status_0.h5
```

后续阶段（preprocess → forward → assess → output → report）均已实现；接口契约由 `database.h5` 和 `status_N.h5` schema 定义（见 `doc/schema.md`）。

## HDF5 files

| File | Lifetime | Contents |
|-----------------|---------------|----------------------------------------------------------------------------------------------------------|
| `database.h5` | Static | Greens at all depths, all freq-band variants, per-module preprocessed data, **paraspace**, config, index |
| `status_{N}.h5` | Per-iteration | Strategy, trials, intermediates, misfits for iteration N |
| `output.h5` | Final | Best-fit parameters, uncertainties, per-phase/station breakdown |

### `/paraspace`

Stores expanded float arrays for parameter-space dimensions:
strike/dip/rake (from grid expansion), depth, frequency, duration（Gaussian STF σ，秒）。
Per-module band selection uses `band_low`/`band_high` in `/config/{ModuleName}/` pointing into
`/paraspace/frequency`. Integer indices live in `/config` (module params) and `/strategy` (search dimensions).
See `doc/schema.md` for details.

## Domain concepts

- **Moment tensor**: 6 components in NED: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
- **Source params**: strike \[0,360), dip [0,90], rake [-90,90] (degrees)
- **Green's functions**: 6-component waveforms per station, pre-computed externally
- **Misfit modules**: XcorrP/S、LagP/S、Psr、PolarityP active。PSR = `abs2(Δlog(rms(S)/rms(P)))`；PolarityP = 归一化 signed amplitude 的 L1 差。RelShift = Aggregate.StdDev composer（基线 config 未注册）。CAP — cancelled.
- **Trial**: one combination of variable params (SDR, depth, frequency, STF duration)
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
