# Implementation & Testing Plan

> **Status:** Phases 1-5 modules ✅ COMPLETE, Phase 6 driver ✅ COMPLETE
> **Remaining:** T15 (forward main.cpp integration), T21-T22 (E2E + cross-language), F1-F4 (verification)
> **Blocked by:** Kokkos not installed — GPU code compiles once Kokkos available

## Phase 1: Foundations (✅ COMPLETE)

### 1.1. Schema & Infrastructure
- [x] Finalize all HDF5 schemas (validate against schema.md)
- [x] Create Julia projects: `input/`, `preprocess/`, `assess/`, `output/` with `Project.toml`
- [x] Create C++ project: `forward/` with `CMakeLists.txt`
- [x] Create shared Julia packages: `MTUtils.jl/`, `AssessUtils.jl/`
- [ ] Write `driver.sh` skeleton (state detection + stage invocation) *(deferred to Phase 6)*

### 1.2. MT Conversion (Cross-Language)
- [x] Implement `MTUtils.jl` — SDR → MT (Julia)
- [x] Implement `mt_utils.h` — SDR → MT (C++)
- [x] Test: cross-language consistency (6 decimal place match)
- [x] Test: boundary conditions (strike=0/360, dip=0/90, rake=-90/90)

### 1.3. HDF5 I/O
- [x] Julia HDF5 reader/writer (for all stages)
- [x] C++ HDF5 reader (forward.cpp — C API)
- [x] C++ HDF5 writer (forward.cpp — C API)
- [x] Test: write in Julia, read in C++ and vice versa

---

## Phase 2: Input + Preprocess Stages

### 2.1. Input Stage — Data Ingestion (runs once)

### 2.1. Waveform Preprocessing (NOT YET IMPLEMENTED)
- [ ] Bandpass filtering (Butterworth, zero-phase)
- [ ] Time-window trimming (wavelength-normalized)
- [ ] Per-module preprocessing: XCorr (synamp), Polarity (gf_pol), PSR (amp matrices)
- [ ] Test: filter frequency content, trim sample indices, synamp identity

### 2.2. Database Writer
- [ ] Write `/config` group (from config.toml)
- [ ] Write `/greens` group (from external GF files)
- [ ] Write `/data` group (preprocessed waveform variants)
- [ ] Write `/index` group (phase index, distance, azimuth)

### 2.3. Preprocess Stage — Trial Generation (runs each loop)
- [ ] Grid expansion from strategy (Cartesian product)
- [ ] Write `/trials` group to `status_0.h5`
- [ ] Write `/strategy` group to `status_0.h5`
- [ ] Test: trial count, trial values, axis-varying logic

### 2.4. Integration
- [ ] First run: raw.h5 + config.toml → database.h5 + status_0.h5 (strategy only, no trials)
- [ ] Loop run: status_{N}.h5 (read strategy) → status_{N}.h5 (write trials)
- [ ] Test: minimal synthetic event end-to-end

---

## Phase 3: Forward Stage

### 3.1. Framework
- [ ] `main()` — argument parsing, stage detection
- [ ] `TrialReader` — read `/trials` from HDF5, SDR→MT pre-conversion to GPU
- [ ] `DataCache` — GPU memory manager, load once, precompute, discard raw
- [ ] `MisfitWriter` — write per-module misfits to HDF5
- [ ] Test: read/write round-trip with synthetic data

### 3.2. XCorr Kernel (Primary)
- [ ] Time-domain precomputation: `CC(obs, GF[:,i])` for i=0..5
- [ ] Per-trial kernel: weighted sum, max-find, normalization → misfit
- [ ] Test: match Julia CPU implementation (100 phases × 100 trials)
- [ ] Test: synamp identity verification (`m'·synamp·m = ‖GF·m‖²`)

### 3.3. Polarity Kernel
- [ ] Precomputation: sum GF over polarity window → `pol_vec[6]`
- [ ] Per-trial: sign of `pol_vec·m` vs observed polarity
- [ ] Test: polarity 0/1 for known matching/mismatching cases

### 3.4. PSR Kernel
- [ ] Precomputation: copy amplitude covariances to GPU
- [ ] Per-trial: compute P/S ratio, compare to observed
- [ ] Test: known P/S ratios produce expected misfits

### 3.5. Integration
- [ ] Full pipeline: load all data, precompute, launch all 3 kernels, write all 3 misfits
- [ ] Verify: no GPU memory leaks, correct output shapes
- [ ] Performance: profile per-trial throughput vs CPU baseline

---

## Phase 4: Assess Stage

### 4.1. Aggregator
- [ ] Per-module masking (XCorr phase-level, Polarity station-level, PSR station-level)
- [ ] Per-module aggregation (sum valid misfits)
- [ ] Module weighting + combination
- [ ] NaN handling (skip masked, error on all-NaN)
- [ ] Test: aggregation matches hand-computed examples, NaN safety

### 4.2. Operator Prompt & Grid Refinement
- [ ] Display best SDR, misfit, current grid to operator
- [ ] Prompt "Continue? [y/N]"
- [ ] On continue: write `status_{N+1}.h5` with `/strategy/converged=0`
- [ ] On break: write `/strategy/converged=1`, `convergence_reason="user"`
- [ ] Compute next grid from current results (center on best, halve steps, 3×3×3 SDR)
- [ ] Depth/frequency subset refinement (20% of best)
- [ ] Test: 2-iteration loop, verify operator prompt flow, grid refinement centers on best trial

### 4.3. Strategy Writer
- [ ] Write updated strategy to `status_{N+1}.h5`
- [ ] Write `/strategy/converged=1` and `convergence_reason="user"` on break
- [ ] Accumulate freq test results and depth misfits

### 4.5. Integration
- [ ] Read status_{N}.h5 → aggregate → prompt operator → write status_{N+1}.h5
- [ ] Test: 2-iteration loop with operator break

---

## Phase 5: Output Stage

### 5.1. Solution Compilation
- [ ] Recompute best fit (verify against assess result)
- [ ] SDR → MT conversion
- [ ] Frequency uncertainty synthesis
- [ ] Depth range synthesis (5% threshold)
- [ ] Test: matches assess best trial, correct uncertainty ranges

### 5.2. Per-Station & Waveforms
- [ ] Per-station misfit breakdown
- [ ] Waveform synthesis (optional, `--waveforms-output`)
- [ ] Summary statistics

### 5.3. Output Writer
- [ ] Write all groups to `output.h5`
- [ ] Test: verify all expected datasets present with correct shapes

---

## Phase 6: Driver & Integration

### 6.1. Driver Script
- [ ] State detection (HDF5 introspection + file existence)
- [ ] Stage invocation in correct order
- [ ] Loop control: read `/strategy/converged`, break to output on converged=1
- [ ] Error reporting

### 6.2. End-to-End Testing
- [ ] Minimal synthetic event (converges in N iterations)
- [ ] Real event (if available)
- [ ] Resume after interruption
- [ ] Batch processing (multiple events)

---

## Deferred (Post-v1)

- [ ] AbsShift kernel (GPU) — **deferred**
- [ ] RelShift kernel (GPU) — **deferred**
- [ ] CAP kernel (GPU) — **deferred**
- [ ] FFT-based XCorr precomputation (cuFFT)
- [ ] Adaptive module weighting in assess
- [ ] Bootstrap uncertainty estimation
- [ ] Visualization / plotting tools