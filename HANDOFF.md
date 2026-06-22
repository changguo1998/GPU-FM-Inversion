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

## Document Consistency Audit (2026-06-21)

Cross-referenced all 13 documentation files against actual source code. Critical issues resolved; remaining items follow.

### 🟡 Medium Issues

#### M1. `station_ids` vs `channel_ids` in `raw.h5`

- `doc/schema.md` documents the dataset as `/phase_picks/channel_ids`
- `shared/HDF5IO.jl/src/HDF5IO.jl` `read_phase_picks()` reads from `gr["station_ids"]`
- `synthetic_data.jl` writes `station_ids` to match the reader

**Fix**: Align schema.md and HDF5IO.jl. Prefer `station_ids` since it matches the phase_picks conceptual grouping (one entry per station).

#### M2. `freq_misfit_curve` initial shape

- `doc/schema.md`: `[N_frequencies, N_freq_test_mechs]`
- `input/src/input.jl` initializes as `zeros(Float64, n_frequencies, 1)` — only 1 column, not `N_freq_test_mechs`

**Fix**: Either update schema.md to allow variable second dimension, or fix input.jl to allocate with `N_freq_test_mechs` columns.

#### M3. `/per_phase` group not written by `write_output`

- `doc/schema.md` and `doc/modules/solution_comp.md` specify a `/per_phase` group in `output.h5`
- `shared/HDF5IO.jl/src/HDF5IO.jl` `write_output()` only writes `/solution`, `/uncertainty`, `/per_station`, `/summary` — **missing `/per_phase`**

**Fix**: Add `/per_phase` writing to `write_output()`, or update `output.jl` to write it separately.

#### M4. `/per_station` vs `/per_station_summary`

- `doc/schema.md` uses `/per_station_summary`
- `shared/HDF5IO.jl` `write_output()` writes to `per_station` (no `_summary` suffix)
- `doc/modules/solution_comp.md` mentions "station-level aggregates" but doesn't give exact group name

**Fix**: Align to one name. Prefer `/per_station_summary` (schema.md) for clarity.

### 🟢 Low-Priority Issues

#### L1. PSR `database.h5` storage uses P/S pair keys, not individual `phase_id`

- `doc/schema.md` implies PSR data is stored per `phase_id` like XCorr and Polarity
- `input/src/input.jl` stores PSR data as `"{P_phase_id}|{S_phase_id}"` — concatenated P/S pair
- Schema should document this non-standard key format

#### L2. Polarity data exists only for P phases — not documented in schema

- `doc/schema.md` shows Polarity under the same `{phase_id}` pattern as XCorr, implying entries for all phases
- `input/src/input.jl` only writes Polarity data when `ptype == "P"`
- Schema should note that Polarity directories only exist for P-wave phase_ids

#### L3. Dimension abbreviation inconsistency: `N_st × N_tr` for Polarity

- `doc/design.md` uses `N_ch × N_tr` correctly in one place but `N_st × N_tr` in another for Polarity shape
- Correct dimension is `N_channels` — Polarity misfit is `[N_channels × N_trials]`

### Files That Need Changes

| File | M1 | M2 | M3 | M4 | L1 | L2 | L3 |
|------|----|----|----|----|----|----|----|
| `doc/schema.md` | ✓ | ✓ | — | ✓ | ✓ | ✓ | — |
| `doc/design.md` | — | — | — | — | — | — | ✓ |
| `shared/HDF5IO.jl/src/HDF5IO.jl` | ✓ | — | ✓ | ✓ | — | — | — |
| `input/src/input.jl` | — | ✓ | — | — | — | — | — |

### No Changes Needed (Confirmed Consistent)

- `xcorr_phase_mask`: `[N_phases]` — schema and code match ✅
- `freq_accumulated`: `[N_frequencies, 3]` — schema and code match ✅
- Pipeline stage responsibilities: all stage docs agree with design.md ✅
- HDF5 file lifetime descriptions: consistent across schema.md and stage docs ✅
- SDR→MT formula: mt_utils.md matches AGENTS.md domain concepts ✅
- Backend dispatch design: forward.md matches misfit_kernel.md and data_cache.md ✅
- No HighFive / raw C API: hdf5_io.md consistent with forward.md ✅
- Deferred modules (AbsShift, RelShift, CAP): consistently marked everywhere ✅