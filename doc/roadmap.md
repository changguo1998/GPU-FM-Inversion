# Roadmap — Focal Mechanism Inversion Pipeline

## Legend

- [x] Completed
- [~] Partial / needs wiring
- [ ] Not started

______________________________________________________________________

## Phase 1: Core Infrastructure (complete)

| Task | Status | Notes |
|---------------------------------------------------------------------|--------|--------------------------------------|
| [x] IO module — HDF5 read/write, type structs, geophysics utilities | Done | `shared/io/` |
| [x] MT module — SDR↔MT conversion (Julia + C++ dual) | Done | Verified cross-language to 1e-12 |
| [x] Grid module — trial generation | Done | `shared/grid/src/trial_gen.jl` |
| [x] Config module — interface declarations | Done | `shared/config/src/Config.jl` |
| [x] Signal module — waveform preprocessing | Done | `shared/signal/` |
| [x] StageLog module — per-stage logging | Done | `shared/stage_log/` |
| [x] Aggregate module — misfit aggregation | Done | `shared/aggregate/` |
| [x] `input.jl` — data ingestion and preprocessing | Done | Writes `database.h5` + `status_0.h5` |
| [x] `preprocess.jl` — trial generation | Done | Reads strategy, writes trials |
| [x] `assess.jl` — weighting, aggregation, grid refinement | Done | Signals via exit code 0/10 |
| [x] `output.jl` — solution compilation | Done | Writes `output.h5` |
| [x] Forward C++ framework — HDF5 I/O, DataCache, kernels | Done | Compiled binary, all test targets |
| [x] Grid refinement logic | Done | `shared/grid/src/grid_refinement.jl` |
| [x] Synthetic test data generator | Done | `tests/synthetic_data.jl` |

## Phase 2: Pipeline Integration (in progress)

| Task | Status | Notes |
|----------------------------------------------|----------------|----------------------------------------------------------------------------------------------------------------|
| [~] `driver.sh` — full 5-stage loop | **Partial** | Input stage works, loop+output behind `exit 0` |
| [ ] `driver.sh` — assess exit code detection | Not started | Need to read exit code 10 |
| [ ] `driver.sh` — resume / skip-input logic | Not started | Detect existing `database.h5` |
| [~] DataCache ↔ database.h5 schema bridge | **Needs work** | DataCache reads `/data/{freq}/{module}/{pid}/` paths; database.h5 stores under `/xcorr/`, `/polarity/`, `/gf/` |
| [ ] Maxlag from database config | Not started | Hardcoded to 50 in `main.cpp` |
| [ ] Waveform synthesis path fix | Not started | `read_greens` uses `greens/` path but DB stores under `/gf/` |

## Phase 3: Testing & Validation

| Task | Status | Notes |
|----------------------------------------|-----------|--------------------------------------------------------|
| [x] MT cross-language consistency test | Done | `tests/test_cross_lang.jl` |
| [x] E2E test script | Done | `tests/test_e2e.sh` (may need update for current code) |
| [ ] End-to-end pipeline test | Not wired | Needs Phase 2 completion first |
| [ ] Forward stage integration test | Not wired | Needs DataCache path bridge |

## Phase 4: Future Modules

| Task | Status | Notes |
|---------------------------------------------|-------------|------------------------------------------------|
| [ ] PSR module pipeline integration | Not started | C++ kernel exists, no data path in database.h5 |
| [ ] AbsShift module | Deferred | |
| [ ] RelShift module | Deferred | |
| [ ] Operator prompt in non-interactive mode | Not started | Currently reads stdin |

## Known Blockers

1. **DataCache path mismatch**: `load_combo()` reads from `/data/{freq_idx}/{module}/{pid}/` paths. The new `database.h5` schema stores data under `/xcorr/obs/{phase}-{band}/`, `/polarity/obs/`, `/gf/{depth}/`. Forward stage cannot find data.
1. **`driver.sh` exit 0**: Pipeline stops after input stage. Loop and output are below `exit 0`.
1. **`read_greens` path**: Uses `greens/{pid}/{depth}` but database stores GF at `/gf/{depth}/{channel_id}`. Waveform synthesis in output.jl silently skips all phases.
