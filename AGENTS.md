# AGENTS.md — Focal Mechanism Inversion (CUDA Rewrite)

## Project identity

CUDA-accelerated focal mechanism inversion pipeline. Julia for preprocessing + strategy, C++ with custom OpenMP/CUDA backend for GPU misfit, HDF5 for data exchange.

Original Julia code removed from tree. Current rewrite: flat stage scripts in `scripts/`, shared Julia packages in `shared/`, C++ CMake project in `forward/`, focused tests.

## Project layout

```
scripts/        Flat stage scripts (input, preprocess, assess, output)
shared/         Julia packages by function (io, mt, grid, signal, aggregate, config, stage_log)
forward/        C++ CMake project (main, hdf5_io, data_cache, mt_utils, kernels, tests)
driver.sh       Bash orchestration
tests/          E2E + cross-language tests
examples/       Data generation scripts
config_sample.jl   Template pipeline configuration
```

## Per-module reference

| Module    | AGENTS.md                    | Role                                         |
|-----------|------------------------------|----------------------------------------------|
| IO        | `shared/io/AGENTS.md`        | HDF5 I/O, type structs, geophysics utilities |
| MT        | `shared/mt/AGENTS.md`        | SDR ↔ MT conversion                          |
| Grid      | `shared/grid/AGENTS.md`      | Trial generation + grid refinement           |
| Signal    | `shared/signal/AGENTS.md`    | Waveform preprocessing                       |
| Aggregate | `shared/aggregate/AGENTS.md` | Misfit aggregation, uncertainty              |
| Config    | `shared/config/AGENTS.md`    | Pipeline configuration interface             |
| StageLog  | `shared/stage_log/AGENTS.md` | Per-stage logging                            |
| Forward   | `forward/AGENTS.md`          | C++ misfit computation                       |

## Pipeline (5 stages)

```
driver.sh: input (once) → loop: [preprocess → forward → assess → [repeat]] → output
```

| Stage      | Language | File                    | Role                                                                           |
|------------|----------|-------------------------|--------------------------------------------------------------------------------|
| input      | Julia    | `scripts/input.jl`      | Read config, preprocess data → `database.h5`; initial strategy → `status_0.h5` |
| preprocess | Julia    | `scripts/preprocess.jl` | Generate trials from strategy → `status_{N}.h5`                                |
| forward    | C++      | `forward/src/main.cpp`  | GPU misfit: per-module, per-phase, per-trial. No weights, no aggregation       |
| assess     | Julia    | `scripts/assess.jl`     | Weights, aggregate, refine grid, prompt operator (exit 0/10 signaling)         |
| output     | Julia    | `scripts/output.jl`     | Compile solution → `output.h5`                                                 |

## HDF5 files (4)

| File            | Lifetime      | Contents                                                                                  |
|-----------------|---------------|-------------------------------------------------------------------------------------------|
| `database.h5`   | Static        | Greens at all depths, all freq-band variants, per-module preprocessed data, config, index |
| `status_{N}.h5` | Per-iteration | Strategy, trials, misfits for iteration N                                                 |
| `output.h5`     | Final         | Best-fit parameters, uncertainties, per-phase/station breakdown                           |

## Domain concepts

- **Moment tensor**: 6 components in NED: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
- **Source params**: strike \[0,360), dip [0,90], rake [-90,90] (degrees)
- **Green's functions**: 6-component waveforms per station, pre-computed externally
- **Misfit modules**: XCorr, Polarity (active). PSR — C++ kernel exists, Julia preprocessing optional. AbsShift, RelShift — deferred. CAP — cancelled.
- **Trial**: one combination of variable params (SDR, depth, frequency, etc.)
- **Phase** = station + channel + wave type (P/S) — channels subsumed by phases
- **Phase key**: `{network}.{station}.{component}.{phase_type}` (e.g. `IU.COLA.00.P`)

## Coding assumptions

These conventions apply across the entire project.

### Architecture

1. `forward.cpp` is stateless — reads data + trials, writes raw misfits. No weights, no aggregation, no strategy knowledge.
1. `assess.jl` owns all strategy: weights, channel selection, grid refinement, operator prompt. Signals continue/converged via exit code (0/10).
1. All frequency-band variants precomputed upfront by `input.jl` — no runtime filtering.
1. Green's functions pre-computed externally — loaded by `input.jl`, never computed by pipeline.
1. Misfit shapes (unweighted): XCorr `[N_ph × N_tr]`, Polarity `[N_st × N_tr]`, PSR `[N_st × N_tr]`. Weights applied in assess.
1. Config bootstrapped via `config.jl` (Julia script implementing `Config` module interface) — only `input.jl` reads it. All config written to `database.h5`; subsequent stages read from HDF5.
1. **Flat scripts**: stage scripts have zero `function` definitions — straight-line top-level execution. No `main()` wrappers.
1. **Shared packages**: utility code lives in `shared/` Julia packages imported via `using`. Each package has own `Project.toml`.

### Implementation patterns

9. **Forward backend**: custom dispatch (`Device<OpenMP>` / `Device<CUDA>`) via template, not Kokkos. Data stored as flat `double*` with explicit strides — no multi-dimensional View abstraction.
1. **Linear XCorr**: CC(obs, GF[:,i]) precomputed on host CPU by `DataCache`. GPU kernel does weighted sum of precomputed CCs — avoids redundant computation per trial.
1. **Grid refinement**: center on best SDR, halve step sizes, fixed 3×3×3 grid, depth/freq subsets within 20% of best misfit.

### Formatting

12. **4-space indent**: all languages. Julia (`.JuliaFormatter.toml` `indent=4`), C++/CUDA (`.clang-format` `IndentWidth: 4`), Bash (`shfmt -i 4`). No tabs.
01. **Space around operators**: spaces on both sides of `=`, `==`, `<`, `>`, `+`, `-`, `*`, etc. C++: `SpaceBeforeAssignmentOperators: true`. Julia: `whitespace_typedefs = true`.
01. **Compact style**: no unnecessary line breaks. Short blocks stay on fewer lines where readable. Functions multi-line (room for docstrings). Loops/conditionals compact.
01. **Short docstrings**: every exported/public function gets 1-3 line docstring. Julia `"""..."""` above definition. C++ `///` or `//` above declaration. What it does, not how.

### Data conventions

16. All angles in degrees in HDF5 and Julia; radians only in C++ forward module internally.
01. All HDF5 datasets `Float64` unless noted. Scalars stored as scalar datasets.
01. TOML config (`config.toml`) is for reference only — pipeline uses Julia `config.jl`.

### Workflow

19. **Format before stage or commit**: run formatter on all changed files before staging. Julia: `julia --project=. -e 'using JuliaFormatter; format(".")'`. C++: `clang-format -i` on changed headers and source files. Unformatted code blocks stage.
01. **Docs follow code**: update or create relevant `doc/` and per-module `AGENTS.md` in same change as code modification, before staging. Stale docs treated as technical debt — no separate "docs PR" later.
01. **Logical commit grouping**: split work into focused, independently-reviewable commits. No lumping unrelated changes (bugfix + refactor + feature) into single commit. Each commit message uses conventional prefix: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `build:`.
01. **Parallelize independent work**: fan out independent tasks across available compute budget — subagents, parallel shell, concurrent pipeline stages. No sequential execution where parallelism safe. Constraint: shared resources (GPUs, HDF5 file locks, mutable state) serialize where necessary.
01. **Caveman communication by default**: use compressed, article-free, fragment-style responses (as defined by `caveman` skill) for all tool-assisted development interactions. Full prose reserved for commit messages, user-facing documentation, and cases where compression creates ambiguity.

### Versioning

24. No CI configured. Tests run manually.
01. HDF5 schema is the API contract between stages — no versioning mechanism yet (schema changes require coordinated stage updates).
01. No formal semantic versioning — package versions are `0.1.0` placeholders.
