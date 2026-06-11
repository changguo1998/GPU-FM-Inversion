# Stage: `setup.jl` — Data Preprocessing & Trial Generation

## Role

Two responsibilities depending on context:
- **First run (iteration 0)**: Read `raw.h5`, preprocess all data → `database.h5`. Generate initial trials → `status_0.h5`.
- **Subsequent runs (iteration N > 0)**: Read strategy from `status_{N}.h5`, generate trials → append to `status_{N}.h5`.

## Inputs

| Source | First Run (N=0) | Subsequent (N>0) |
|--------|-----------------|-------------------|
| `raw.h5` | Read | — |
| `config.toml` | Read | — |
| `database.h5` | Write (new) | Read-only (for validation) |
| `status_{N}.h5` | Write `/strategy` + `/trials` (new) | Read `/strategy`, write `/trials` |

## Outputs

- **database.h5**: All preprocessed data (written once)
- **status_{N}.h5**: Trials + strategy written into existing file

## Responsibilities

1. **Preprocess raw data**: filter waveforms to frequency bands, trim time windows, extract module-specific preprocessing output, store in `database.h5`
2. **Load Green's functions**: read external GF files, store by phase × depth in `database.h5`
3. **Write algorithm config**: parse `config.toml`, write into `database.h5` and `status_0.h5`
4. **Generate trials**: expand grid (Cartesian product of varying axes) → write `/trials` group
5. **Write strategy**: initial or updated search grid into `status_{N}.h5`

## Tool Stack

- Julia (`HDF5.jl`, `DSP.jl` for filtering, `Dates.jl`, `TOML.jl`)
- Butterworth bandpass filter (DSP.jl, zero-phase forward-backward)
- Time-window trimming
- Cartesian product iterator for trial generation

## What It Does NOT Do

- Does NOT modify `database.h5` on subsequent runs
- Does NOT compute misfits (that's `forward.cpp`)
- Does NOT apply weights or make strategy decisions (that's `assess.jl`)