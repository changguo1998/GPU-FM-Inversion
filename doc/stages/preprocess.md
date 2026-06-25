# Stage: `scripts/preprocess.jl` — Trial Generation

## Role

Runs on every loop iteration. Reads the current search strategy from `status_{N}.h5`, generates trials via `shared/grid/` (Cartesian product of varying axes), and writes the `/trials` group into `status_{N}.h5`.

This is the per-loop portion of the former `setup.jl`. It is **not** responsible for data preprocessing (filtering, trimming, GF loading) — that is `input.jl`, which runs once.

## Inputs

| Source          | Description                                                                              |
|-----------------|------------------------------------------------------------------------------------------|
| `status_{N}.h5` | Reads `/strategy` — current grid parameters (SDR center, step sizes, depth/freq indices) |
| `database.h5`   | Read-only (for validation / config reference)                                            |

## Outputs

| Source          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `status_{N}.h5` | Writes `/trials` — Cartesian product of varying axes expanded from strategy |

## Responsibilities

1. **Read strategy**: load current grid parameters from `status_{N}.h5`
1. **Expand grid**: Cartesian product of strike, dip, rake, depth indices, freq indices
1. **Write trials**: store expanded trial table in `/trials` group of `status_{N}.h5`

## Tool Stack

- Julia (`HDF5.jl`)
- Cartesian product iterator for trial generation

## What It Does NOT Do

- Does NOT preprocess raw data (that's `input.jl`)
- Does NOT modify `database.h5`
- Does NOT compute misfits (that's `forward.cpp`)
- Does NOT apply weights or make strategy decisions (that's `assess.jl`)
