# Parallelism Analysis for CUDA Rewrite

## Current parallelism model

All parallelism uses Julia's `Threads.@threads` (CPU multi-threading). The CUDA rewrite should target the same computational hotspots.

## Identified parallelism points

### 1. Green's function computation (`calcgreen!()` in mathematics.jl:81-95)

**Current**: `Threads.@threads for i in idxlist` — one thread per unique station.
**Work per station**: Compute DWN waveform (`DWN.dwn()` + `freqspec2timeseries()`) for a single station — includes frequency-domain computation, FFT, and Green's function assembly.
**CUDA opportunity**: DWN computation per station is independent but involves frequency-domain loops — could batch multiple stations or parallelize the DWN kernel itself. The DWN computation is the main bottleneck here.
**Data**: Station metadata (distance, azimuth, depth), velocity model, source parameters.

### 2. Grid search misfit evaluation (`inverse!()` in mathematics.jl:175-181)

```
Threads.@threads for i = 1:(length(newsdr)*Lp)
    (p, q) = divrem(i - 1, Lp)
    newmisfitdetail[p, q] = misfit_func(phase[q], moment_tensor[p])
end
```

**This is the hottest loop** — it evaluates misfit for ALL (strike,dip,rake) × ALL (phase,station) combinations.

**Grid sizes**:
- Stage 1, step 1: ~(33×8×37) = ~9,768 SDR points × N phases
- Stage 1, step 2: ~(21×7×13) = ~1,911 SDR points (centered on previous)
- Stage 1, step 3: ~(5×3×3) = ~45 SDR points (centered on previous)
- Frequency test: N_iterations × small grids (5×3×3 typically)
- Depth refinement: N_depths × Stage 1 search

**CUDA opportunity**: PERFECT for GPU — each (SDR, phase) pair is independent. Launch as a 2D grid where:
- Block x-dim = Lp (phase-station pairs, typically 10-100)
- Block y-dim = len(newsdr) (SDR parameter combinations, ~10-10,000)
- Each thread evaluates one misfit function

**Data dependencies**:
- `phase` dict: contains preprocessed observed record + Green's function matrix (read-only during search)
- `moment_tensor`: 6-element vector (computed from SDR, read-only)
- Output: scalar misfit value per (SDR, phase) pair → reduced via weighted sum on CPU

### 3. Cross-correlation computation (`_xcorr()` in XCorr.jl:23-49)

```
for i in axes(r, 1)        # 2*maxlag+1 lags (~100-1000)
    for j in axes(r, 2)    # Wu*Wv combinations (1 or 6)
        for l = 0:(maxv-minv)   # signal length (~200-2000)
            r[i,j] += u[minu+l] * v[minv+l]
        end
    end
end
```

**CUDA opportunity**: This is essentially a 2D cross-correlation — well-suited for GPU via:
- FFT-based cross-correlation (cuFFT): O(N log N) instead of O(N²)
- Direct convolution with shared memory for small lags

**This is duplicated in**: AbsShift.jl, RelShift.jl, and CAP.jl (each has its own `_xcorr()`).

### 4. DTW error map computation (DTW.jl:55-64)

```
for i = 1:N, j = 1:2*maxlag+1
    e[i,j] = (observed[i] - synthetic[i+j-lag])²
end
```

**CUDA opportunity**: Element-wise operations on a 2D grid — trivial to parallelize. The subsequent cumulation and backtracking are sequential but cheap (O(N×maxlag) which is small).

### 5. Velocity model interpolation (`frameinterp!()` in semmodel.jl:78-96)

```
Threads.@threads for idx in CartesianIndices(othermesh)
    # Linear interpolation along one dimension
end
```

**CUDA opportunity**: Element-wise interpolation on a 3D grid — straightforward GPU kernel.

### 6. NLLOC model resampling (`resample_nearest()` in NLLOCshell.jl:80-95)

```
Threads.@threads for idx in CartesianIndices(index)
    # Nearest-neighbor resampling
end
```

**CUDA opportunity**: Independent per-voxel — trivial to parallelize.

## CUDA rewrite strategy

### Phase 1: Grid search (highest impact)
Move `inverse!()` to GPU:
1. Pre-allocate phase data (observed records, Green's function matrices) on GPU
2. Pre-compute all moment tensors for all SDR combinations
3. Launch CUDA kernel: one block per SDR, threads within block per phase
4. Reduce per-SDR weighted misfit on GPU
5. Transfer final misfit array back to CPU for argmin

### Phase 2: Cross-correlation
Replace direct O(N²) convolution with cuFFT-based cross-correlation via `cufftExecC2C` or use GPU-accelerated direct convolution with tiling in shared memory.

### Phase 3: Green's function computation
Batch DWN computations or parallelize DWN internal loops. This depends on DWN.jl internals which need separate analysis.

### Phase 4: Preprocessing
Move filter, resample, trim operations to GPU using cuDSP/cuFFT for batched signal processing across all stations.

## Data transfer considerations

**Read-only data on GPU** (upload once per inversion):
- All Green's function matrices (Float32, ~6 × N_samples × N_stations)
- All observed records (Float32, ~N_samples × N_stations)
- Phase metadata (filter bands, trim windows, lags)

**Computed per-iteration**:
- Moment tensors (6 × N_sdr, Float32) — cheap on GPU

**Output** (download after each grid search):
- Misfit detail matrix (N_sdr × N_phases, Float32)
- Weighted misfit vector (N_sdr, Float32)

## Memory estimate

For a typical event with 20 stations × 3 components = 60 phases, Green's functions at ~1000 samples:
- Green's functions: 60 × 1000 × 6 × 4 bytes = ~1.4 MB
- Observed records: 60 × 1000 × 4 bytes = ~240 KB
- Grid search (Stage 1, step 1): ~10,000 SDR × 60 phases × 4 bytes = ~2.4 MB

Total GPU memory: ~5-10 MB — easily fits in any modern GPU.