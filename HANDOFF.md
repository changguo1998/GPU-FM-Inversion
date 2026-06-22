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
- No cuFFT, no CAP, no FFT-based XCorr — cancelled. AbsShift, RelShift — deferred

---

## Doc-Code Consistency

Completed 2026-06-22. All 7 issues (M1–M4, L1–L3) fixed in commit `6ded2ec`.
