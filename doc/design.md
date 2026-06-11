# Design: CUDA-Accelerated Focal Mechanism Inversion

## Overview

CUDA-accelerated pipeline for determining earthquake focal mechanisms through iterative grid search. A 4-stage loop with GPU-accelerated misfit computation and Julia-based preprocessing, strategy, and export.

## Stage Partitioning

```
setup → forward → assess → [repeat] → export
```

| Stage | Language | Responsibility |
|-------|----------|----------------|
| `setup.jl` | Julia | Data preprocessing → `database.h5`; trial generation → `status_{N}.h5` |
| `forward.cpp` | C++/Kokkos | GPU misfit computation: per-module, per-phase, per-trial. Stateless. No weights. |
| `assess.jl` | Julia | Weighting, aggregation, grid refinement, operator prompt → `status_{N+1}.h5` |
| `export.jl` | Julia | Compile final solution → `output.h5` |
| `driver.sh` | Bash | Stateless orchestration: file-state detection, stage invocation, loop control |

Orchestration detail: `doc/stages/`

## Control Flow

```
driver.sh loops:
  1. setup.jl    → database.h5 (once), status_0.h5 (first run)
                   → status_{N}.h5 trials (subsequent runs, reads strategy from status_{N}.h5)
  2. forward.cpp → reads database.h5 + status_{N}.h5 /trials
                   → writes status_{N}.h5 /misfits
   3. assess.jl   → reads status_{N}.h5 /trials + /misfits
                    → writes status_{N+1}.h5 /strategy (refined grid)
                    → prompts operator for continue/break
                    → sets /strategy/converged=1 on break
   4. if converged=0 → loop back to setup; if converged=1 → export.jl → output.h5
```

## Data Flow (Stage Level)

```
raw.h5 (external) ──┐
                     │
config.toml ────► setup.jl ──► database.h5 (static, all preprocessed data)
                     │          status_0.h5 (initial strategy + trials)
                     │
              status_{N}.h5 ◄──────┐
                     │              │
                     ▼              │
              forward.cpp ──────────┘  (reads trials, writes misfits)
                     │
                     ▼
              assess.jl
                     │
                     ▼
              status_{N+1}.h5  (new strategy for next iteration)
                     │
              ┌──────┴──────┐
              │             │
         converged?    loop back to setup
              │
              ▼
         export.jl
              │
              ▼
         output.h5
```

## Data Files

| File | Lifetime | Produced By | Contents |
|------|----------|-------------|----------|
| `raw.h5` | Static (input) | External | Event info, station metadata, raw SAC waveforms |
| `database.h5` | Static | `setup.jl` (first run) | All preprocessed data: Greens at all depths, filtered waveform variants, per-module preprocessing, algorithm config |
| `status_{N}.h5` | Per-iteration | `setup.jl` (strategy + trials), `forward.cpp` (misfits), `assess.jl` (strategy for N+1) | Self-contained: strategy, trials, misfits for iteration N |
| `output.h5` | Final | `export.jl` | Best-fit parameters, uncertainties, per-station breakdown, synthetic waveforms |
| `config.toml` | Bootstrap | User | Misfit module list, frequency bands, depth range, initial grid params |

## Key Design Rules

1. **`forward.cpp` is stateless** — reads preprocessed data + trial params, writes raw misfits. No weighting, no aggregation, no strategy knowledge.
2. **`assess.jl` owns all strategy** — weights, phase selection, grid refinement, and operator prompt for continue/break.
3. **All frequency-band variants precomputed upfront** in `database.h5`. No runtime filtering.
4. **Misfits are unweighted** — weights applied in assess. XCorr: `[N_ph × N_tr]`, Polarity: `[N_st × N_tr]`, PSR: `[N_st × N_tr]`.
5. **Green's functions pre-computed externally** — loaded by `setup.jl`, never computed by the pipeline.
6. **Linear decomposition** — cross-correlation precomputed on GPU: `CC(obs, GF[:,i])` for i=0..5. Per-trial: weighted sum of precomputed CCs.
7. **Dynamic grid refinement** — `assess.jl` refines grid from results each iteration, centered on best trial.

## Dimension Symbols

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `N_stations` | Stations | 10–30 |
| `N_phases` | Phase-station pairs (P+S) | 20–60 |
| `N_depths` | Depth levels for Greens | 10–40 |
| `N_frequencies` | Frequency band combinations | configurable |
| `N_modules` | Active misfit modules | 3 (XCorr, Polarity, PSR) |
| `N_trials` | Trials per iteration | 10–100000 |
