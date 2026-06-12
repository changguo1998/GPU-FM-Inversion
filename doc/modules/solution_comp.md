# Module: Solution Compilation & Uncertainty

## Purpose

Compile final solution from converged pipeline results: best-fit parameters, uncertainties, per-station breakdown.

## Used By

- `output.jl`

## Operations

### 1. Best-Fit Verification

Re-aggregate misfits (using `Aggregator` module) and verify best trial matches `/strategy/best_sdr`.

### 2. SDR → MT Conversion

Convert best SDR to moment tensor using `MTUtils`.

### 3. Frequency Uncertainty

Read `freq_accumulated` from strategy → compute std of the best SDR values across frequency bands. `freq_misfit_curve` is the separate source for frequency-band misfit reporting.

### 4. Depth Range

Apply 5% tolerance to `depth_misfit_accumulated`:
- Find depth with minimum misfit
- Include all depths within 5% of minimum
- Return `[min_depth, max_depth]`

### 5. Per-Station Breakdown

Extract per-module misfit at best trial for each phase, with station identification.

### 6. Waveform Synthesis (Optional)

Compute `synthetic = GF × best_MT` for visual QC.

## Output

All groups in `output.h5`:
- `/solution`: best-fit SDR, MT, misfit
- `/uncertainty`: SDR std, depth range, freq misfit curve
- `/per_station`: misfit per module per phase, selection mask, CC values
- `/summary`: total iterations, total trials, convergence reason
- `/waveforms` (optional): synthetic seismograms

## Testing Strategy

- Best-fit: matches assess.jl result exactly
- Depth range: 5% threshold applied correctly
- Frequency uncertainty: std computation on known vectors
- Waveform synthesis: `GF × MT` matches manual computation
