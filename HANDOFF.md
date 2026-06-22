# HANDOFF — Focal Mechanism Inversion (CUDA Rewrite)

## Pipeline Implementation Status: Complete ✅

All tasks (T1–T22) are complete. Working tree clean. E2E test passes.

Milestones completed:

- **mi-01**: Foundation — MTUtils, HDF5IO, build files, synthetic data
- **mi-02**: Input stage — waveform preprocessing, input integration
- **mi-03**: Preprocess stage — trial generation, preprocess integration
- **mi-04**: Assess stage — misfit aggregation, grid refinement, assess integration
- **mi-05**: Output stage — solution compilation, output integration
- **mi-06**: Forward stage — GPU data cache, XCorr/Polarity/PSR kernels, main.cpp
- **mi-07**: Orchestration & testing — driver.sh, E2E test, cross-language suite

## Not Done

- **C++ compilation**: Kokkos not installed; all C++ code complete but not compiled/tested

## Key Constraints Preserved

- `forward.cpp` stateless — no weights, no aggregation
- `assess.jl` owns all strategy — grid refinement, operator prompt
- All frequency-band variants precomputed in `database.h5` by `input.jl`
- Misfits unweighted per-module shapes
- Raw HDF5 C API only (no HighFive)
- No cuFFT, no CAP/AbsShift/RelShift (deferred)

---

## Document Consistency Audit (2026-06-22 — all M*/L* resolved)

Cross-referenced all 13 documentation files against actual source code.
All 7 issues (M1–M4, L1–L3) resolved in commit below.

### 🟡 Medium Issues (all resolved 2026-06-22)

#### M1. `station_ids` vs `channel_ids` in `raw.h5` ✅ No change needed

Schema, HDF5IO reader, and synthetic data all use `station_ids`. Already consistent.

#### M2. `freq_misfit_curve` initial shape ✅ Fixed

`input/src/input.jl`: `n_frequencies × 1` → `n_frequencies × freq_test_max_iter`

#### M3. `/per_phase` group not written by `write_output` ✅ Fixed

Added `/per_phase` group to `HDF5IO.write_output()`; `solution_comp.jl` now builds separate `per_phase` and `per_station_summary` dicts.

#### M4. `/per_station` vs `/per_station_summary` ✅ Fixed

Renamed to `/per_station_summary` everywhere (schema, HDF5IO, output.jl).
`solution_comp.jl` now computes station-level aggregates (n_channels, n_phases, mean_cc, polarity_match, misfit_total).

### 🟢 Low-Priority Issues (all resolved 2026-06-22)

#### L1. PSR P/S pair keys ✅ Documented

Added note in `doc/schema.md`: PSR stored as `"{P_phase_id}|{S_phase_id}"` pair keys.

#### L2. Polarity data only for P phases ✅ Documented

Added "Only written for P-wave phase_ids" annotation to Polarity datasets in `doc/schema.md`.

#### L3. Dimension `N_st × N_tr` ✅ Already correct

`doc/design.md` consistently uses `N_channels × N_trials`. No change needed.

### Changed Files

| File | M2 | M3 | M4 | L1 | L2 |
|------|----|----|----|----|----|
| `doc/schema.md` | — | — | — | ✓ | ✓ |
| `doc/modules/hdf5_io.md` | — | — | ✓ | — | — |
| `input/src/input.jl` | ✓ | — | — | — | — |
| `shared/HDF5IO.jl/src/HDF5IO.jl` | — | ✓ | ✓ | — | — |
| `output/src/output.jl` | — | ✓ | ✓ | — | — |
| `output/src/solution_comp.jl` | — | ✓ | ✓ | — | — |
| `output/test/runtests.jl` | — | ✓ | ✓ | — | — |
| `output/test/test_output_stage.jl` | — | ✓ | ✓ | — | — |
| `shared/HDF5IO.jl/test/runtests.jl` | — | ✓ | ✓ | — | — |
| `tests/test_e2e.jl` | — | ✓ | ✓ | — | — |

### No Changes Needed (Confirmed Consistent)

- `xcorr_phase_mask`: `[N_phases]` — schema and code match ✅
- `freq_accumulated`: `[N_frequencies, 3]` — schema and code match ✅
- Pipeline stage responsibilities: all stage docs agree with design.md ✅
- HDF5 file lifetime descriptions: consistent across schema.md and stage docs ✅
- SDR→MT formula: mt_utils.md matches AGENTS.md domain concepts ✅
- Backend dispatch design: forward.md matches misfit_kernel.md and data_cache.md ✅
- No HighFive / raw C API: hdf5_io.md consistent with forward.md ✅
- Deferred modules (AbsShift, RelShift, CAP): consistently marked everywhere ✅
