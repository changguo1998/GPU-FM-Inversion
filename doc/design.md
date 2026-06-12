# Design: CUDA-Accelerated Focal Mechanism Inversion

## Overview

CUDA-accelerated pipeline for determining earthquake focal mechanisms through iterative grid search. A 5-stage pipeline: one-time initialization, then a 3-stage loop with GPU-accelerated misfit computation and Julia-based preprocessing, strategy, and output compilation.

## Stage Partitioning

```
input (once) ──→ loop: [preprocess → forward → assess → [repeat]] ──→ output
```

| Stage | Language | Runs | Responsibility |
|-------|----------|------|----------------|
| `input.jl` | Julia | Once (before loop) | Data ingestion → `database.h5`; initial strategy → `status_0.h5` |
| `preprocess.jl` | Julia | Each loop | Trial generation from strategy → `status_{N}.h5` |
| `forward.cpp` | C++/Kokkos | Each loop | GPU misfit computation: per-module, per-phase, per-trial. Stateless. No weights. |
| `assess.jl` | Julia | Each loop | Weighting, aggregation, grid refinement, operator prompt → `status_{N+1}.h5` |
| `output.jl` | Julia | Once (after loop) | Compile final solution → `output.h5` |
| `driver.sh` | Bash | Entire run | Stateless orchestration: file-state detection, stage invocation, loop control |

Orchestration detail: `doc/stages/`

## Control Flow

```
driver.sh:
  1. input.jl (once) → database.h5 + status_0.h5 (initial strategy, no trials yet)
  2. loop:
     a. preprocess.jl   → reads status_{N}.h5 /strategy
                          → writes status_{N}.h5 /trials
     b. forward.cpp     → reads database.h5 + status_{N}.h5 /trials
                          → writes status_{N}.h5 /misfits
     c. assess.jl       → reads status_{N}.h5 /trials + /misfits
                          → writes status_{N+1}.h5 /strategy (refined grid)
                          → prompts operator for continue/break
                          → sets /strategy/converged=1 on break
     d. if converged=0 → loop back to step 2a (next N)
        if converged=1 → break to step 3
  3. output.jl → output.h5
```

## Data Flow (Stage Level)

```
raw.h5 (external) ──┐
                     │
config.toml ────► input.jl (once) ──► database.h5 (static, all preprocessed data)
                                         status_0.h5 (initial strategy, no trials yet)
                                              │
                        ◄─────────────────────┘
                        │
                   preprocess.jl (each loop) ──► reads strategy, writes trials
                        │
                        ▼
                 status_{N}.h5 ◄──────┐  (trials added by preprocess)
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
            converged?    loop back to preprocess
                 │
                 ▼
            output.jl ──► output.h5
```

## Data Files

| File | Lifetime | Produced By | Contents |
|------|----------|-------------|----------|
| `raw.h5` | Static (input) | External | Event info, station metadata, raw SAC waveforms |
| `database.h5` | Static | `input.jl` (first run, once) | All preprocessed data: Greens at all depths, filtered waveform variants, per-module preprocessing, algorithm config |
| `status_{N}.h5` | Per-iteration | `input.jl` (initial strategy), `preprocess.jl` (trials), `forward.cpp` (misfits), `assess.jl` (strategy for N+1) | Self-contained: strategy, trials, misfits for iteration N |
| `output.h5` | Final | `output.jl` | Best-fit parameters, uncertainties, per-station breakdown, synthetic waveforms |
| `config.toml` | Bootstrap | User | Misfit module list, frequency bands, depth range, initial grid params |

## Key Design Rules

1. **`forward.cpp` is stateless** — reads preprocessed data + trial params, writes raw misfits. No weighting, no aggregation, no strategy knowledge.
2. **`assess.jl` owns all strategy** — weights, phase selection, grid refinement, and operator prompt for continue/break.
3. **All frequency-band variants precomputed upfront** in `database.h5`. No runtime filtering.
4. **Misfits are unweighted** — weights applied in assess. XCorr: `[N_ph × N_tr]`, Polarity: `[N_st × N_tr]`, PSR: `[N_st × N_tr]`.
5. **Green's functions pre-computed externally** — loaded by `input.jl`, never computed by the pipeline.
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
