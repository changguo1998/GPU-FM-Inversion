# AGENTS.md — Focal Mechanism Inversion (从头开发)

## Project identity

震源机制反演管道。Julia 数据接入 + 预处理，HDF5 数据交换。当前为从头开发第一阶段，仅完成数据接入与初始化。

## Project layout

```
scripts/        Flat stage scripts (当前仅 input.jl)
shared/         Julia packages by function (io, mt, grid, signal, aggregate, config, stage_log)
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

## 当前阶段

仅完成 `input.jl` — 数据接入与初始化阶段。

```
scripts/input.jl  (一次) → database.h5 + status_0.h5
```

后续阶段（preprocess → forward → assess → output）待开发，接口契约由 `database.h5` 和 `status_N.h5` schema 定义（见 `doc/schema.md`）。

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
- **Misfit modules**: XCorr, Polarity (active). PSR, AbsShift, RelShift — deferred. CAP — cancelled.
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
1. **Flat scripts**: stage scripts have zero `function` definitions — straight-line top-level execution. No `main()` wrappers.
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
