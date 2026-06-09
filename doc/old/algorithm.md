# Algorithm Details

## SDR → Moment Tensor Conversion

`dc2ts(sdr)` in `mathematics.jl:67-78` converts double-couple parameters to a 6-component moment tensor in NED coordinate system:

```
Mxx = -[sin(2s)·sin(d)·cos(r) + sin²(s)·sin(2d)·sin(r)]
Myy = sin(2s)·sin(d)·cos(r) - cos²(s)·sin(2d)·sin(r)
Mzz = sin(2d)·sin(r)
Mxy = cos(2s)·sin(d)·cos(r) + 0.5·sin(2s)·sin(2d)·sin(r)
Mxz = -[cos(s)·cos(d)·cos(r) + sin(s)·cos(2d)·sin(r)]
Myz = -[sin(s)·cos(d)·cos(r) - cos(s)·cos(2d)·sin(r)]
```

Where s=strike(rad), d=dip(rad), r=rake(rad).

## Grid Search (searchingMethod/Grid.jl)

Parameter space: strike ∈ [0, 360), dip ∈ [0, 90], rake ∈ [-90, 90]

Three-stage coarse-to-fine refinement:
```
Stage 1:  Step=(11°, 13°, 14°) → Step=(5°, 3°, 3°) → Step=(1°, 1°, 1°)
Stage 2+: Center on previous result, search ±2·step in each direction
```

`Grid.continueloop!()` returns false after first iteration (single-pass grid search).
`Grid.newparameters()` generates all (strike,dip,rake) combinations from current bounds.
`Grid.setstart!/setstep!/setstop!()` update bounds between iterations.

## Core Inversion Loop (`inverse!()` in mathematics.jl:156-188)

```
INPUT: env, modules (misfit functions), searchingMethod (Grid)
OUTPUT: (sdr, phaselist, misfit, misfitdetail)

1. Build phaselist: all (module, phase) pairs across stations
2. Build weightvec: per-pair weights
3. LOOP while searchingMethod.continueloop!():
   a. Generate new SDR parameters via Grid.newparameters()
   b. Convert all SDRs to moment tensors via dc2ts()
   c. PARALLEL: for each (SDR, phase) pair:
        misfit[phase_index](phase, moment_tensor[sdr_index])
   d. Weighted sum: misfit = misfitdetail × weightvec
4. Return all SDRs, phases, misfits, misfit details
```

Parallelism: `Threads.@threads for i = 1:(length(newsdr)*Lp)` — flattened 2D loop over all parameter×phase-station combinations.

## Misfit Functions

### XCorr (Cross-Correlation)

**Preprocess**: Resample observed + Green's function to `xcorr_dt`, bandpass filter (Butterworth), trim time window, normalize.

**Misfit**: 
```
xcorr = cross_correlate(observed, synthetic[:,1:6] × moment_tensor)
misfit = 1.0 - max(|xcorr|)  # Best correlation across all lags
```
Cross-correlation computed via direct O(N²) time-domain convolution in `_xcorr()`.

### Polarity

**Preprocess**: Sum Green's function over P-wave polarity window → 6-component polarity vector.

**Misfit**: 
```
syn_pol = sign(Σ polarity_vector[i] × m[i] for i=1:6)
misfit = (syn_pol == obs_pol) ? 0.0 : 1.0
```
Only applies to P-phase. Observation must be -1, 0, or +1.

### PSR (P/S Amplitude Ratio)

**Preprocess**: Compute observed P/S amplitude ratio (dB), compute Green's function amplitude matrices (6×6 covariance).

**Misfit**: 
```
syn_amp_p = m' × Amp_p_matrix × m
syn_amp_s = m' × Amp_s_matrix × m
misfit = |obs_psr - 10×log10(syn_amp_p/syn_amp_s)|
```

### DTW (Dynamic Time Warping)

**Preprocess**: Resample, bandpass filter, trim observed + Greens (same as XCorr).

**Misfit**: 
```
Build error map: e[i,j] = (observed[i] - synthetic[i+j-lag])²
Cumulate with Sakoe-Chiba band constraint
Backtrack optimal warping path
misfit = cumulative error along path
```
Constraint parameters: `dtw_maxlag` (time shift limit), `dtw_klim` (local slope constraint).

### AbsShift (Absolute Time Shift)

Like XCorr but allows each component (E,N,Z) to shift independently.
Cross-correlation computed per-component triplet, best shift per component.

### RelShift (Relative Time Shift)

All three components (E,N,Z) share the same time shift.
Cross-correlation computed on concatenated 3-component record.

### CAP (Cut-And-Paste)

Zhu & Helmberger (1996) method. Splits Pnl (P-wave + coda) and surface wave portions.

**Preprocess**: Rotate horizontal components (E,N) → (R,T) using station azimuth. Filter Pnl band and surface wave band separately. Trim windows.

**Misfit**: 
```
For each window (Pnl_R, Pnl_Z, Surface_R, Surface_T, Surface_Z):
  syn = synthetic × M0 × moment_tensor
  misfit += weight × ||observed - synthetic||²
M0 = sqrt(Σ||obs||² / Σ||syn||²)  # Moment magnitude scaling
```

## Multi-Stage Workflow (multistage_lib.jl)

`inverse_focalmech!(env, misfits)`:
1. `loadtp!(env)`: Load travel times (tp, ts) from Green's function files
2. `preprocess!(env, misfits)`: Run each module's preprocess! for each phase
3. Three coarse-to-fine grid steps: (11,13,14) → (5,3,3) → (1,1,1)
4. After each step: check convergence (result within 2×step of previous)
5. Return best (strike, dip, rake), misfit values

`inverse_depth(depths, nenv, misfits)`:
1. For each depth: update Green's function paths, run inverse_focalmech!()
2. Return misfit values for all depths

## Frequency Test (inverse.jl:63-103)

Bootstrap-style uncertainty estimation:
```
1. Start from best mechanism (mech)
2. For each iteration:
   a. Generate random perturbation rmech ~ N(0,1), clipped to [-3,3]
   b. New mech = mech + rmech, with bounds: strike[0,360), dip[0,90], rake[-90,90]
   c. Update filter bands for P/S based on new mechanism
   d. Run inverse_focalmech!()
   e. Accumulate results
3. Converge when:
   - strike circular std < sin(5°) AND
   - max(dip_std, rake_std) < 2.5°
   OR iterations > frequency_test_maximum_iteration
```

## Depth Refinement (inverse.jl:106-126, 152-171)

Grid search over depth, bounded by glib model limits:
```
Stage 2: [step2_min_depth, step2_max_depth], step = step2_d_depth
  Search radius = step2_depth_radius
  Iterate until depth change < step2_stop_iterate_when_depth_change_less_than

Stage 3: [step3_min_depth, step3_max_depth] (tighter bounds)
  Same iteration logic
```

## Channel Reselection (inverse.jl:133-149)

After preliminary mechanism is found, filter stations/channels:
1. Compute cross-correlation for each phase
2. Try thresholds 0.0, 0.1, ..., 1.0
3. Select threshold that retains enough stations:
   - <3 stations total → keep all
   - <7 stations total → keep >3
   - ≥7 stations total → keep ≥6

## Signal Processing

### STA/LTA picker (mathematics.jl:13-28)
`pick_stalta(x, ws, wl)`: Short-term/long-term average ratio for arrival picking.
Uses `mean(abs, x)` as processing function.

### Freedom picker (mathematics.jl:30-52)
`pick_freedom(x, wl)`: Uses SVD to detect change in signal dimensionality.
`freedom = Σ(singular_values) / max(singular_value)` — minimum indicates arrival.

### Window ratio picker (mathematics.jl:1-9)
`pick_windowratio(x, wl)`: `std(x[i+wl:i+2wl]) / std(x[i:i+wl])` — maximum indicates change.