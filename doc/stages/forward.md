# Stage: `forward` (C++) — Raw Intermediate Computation

## Role

Runs once per iteration (preprocess → **forward** → assess). Reads the trial
set from `status_N.h5` and the preprocessed reductions from `database.h5`,
runs the misfit kernels per (freq, depth, duration) combo, and writes RAW INTERMEDIATE
PRODUCTS to `status_N.h5:/intermediates/`. Final misfit values (extract /
compose) are produced by `assess.jl` — this stage never writes `/misfits`.

## Usage

```bash
forward/build/forward [--backend auto|cpu|cuda] [--cuda-batch-trials N] \
    <database.h5> <status_N.h5>
```

环境变量 `FM_FORWARD_BACKEND`、`FM_CUDA_BATCH_TRIALS` 提供相同默认值，CLI 优先。
`driver.sh` 可用 `FM_FORWARD_EXE` 选择 binary。Exit code 0 on success。

### Backend 选择

| 请求 | CUDA 可用 | 结果 |
| `cpu` | 任意 | OpenMP CPU |
| `cuda` | 是 | CUDA |
| `cuda` | 否/未编译 | 失败 |
| `auto` | 是 | CUDA |
| `auto` | 无 device/未编译 | OpenMP CPU |
| `auto` | 其他 CUDA 初始化错误 | 失败，不回退 |

CPU-only build 不启用 CUDA language、不链接 cudart。CUDA build 仍可强制 `cpu`。
`--cuda-batch-trials` 仅在最终选择 CUDA 时合法。

## Inputs

| File | Access | Used for |
|---------------|------------|----------------------------------------------------------------------------------|
| `database.h5` | read-only | `/paraspace` axis values, `/config/{Module}`, `/XcorrP|XcorrS/obs|gf`,`/station` |
| `status_N.h5` | read-write | `/trials` (read), `/intermediates/*` (write) |

### Lag half-width is derived from config (not hardcoded)

`maxlag = round(max_lag_periods / band_high_freq / dt)`, with

- `max_lag_periods` from the common validated XcorrP/XcorrS config
- `band_high_freq` = `/paraspace/frequency[band_high - 1]`
- `dt` from `/station/dt`

and then clamped per-combo to `(window_len − 1) ÷ 2` inside `DataCache`
(`entry.xcorr.maxlag`). This mirrors `Misfit.Xcorr.preprocess` exactly — there
is no silent truncation relative to the stored 301-lag reductions (2026-08-14).

## XCorr kernel math

For phase `p` and trial `t` (MT `m = sdr_to_mt(strike, dip, rake)`):

```
dot_lag[c]   = Σ_t obs[t+lag]·gf[t,c]                 (obs shifted +lag, zero-padded in window)
synamp_lag   = Σ_t gf[t−lag,a]·gf[t−lag,b]            (GF shifted −lag, same-shift normalization)
cc_norm[lag] = (mᵀ·dot_lag) / sqrt(obs_n2 · mᵀ·synamp_lag·m)
best_lag     = argmax_lag cc_norm[lag] − maxlag       (Int32, samples, relative shift)
cc_max       = max_lag cc_norm[lag]                     (signed Float64)
```

每个 `(phase, trial)` work item 在 lag 循环前预计算
`mᵀAm` 的 21 个上三角系数，lag 内只执行 21 项累加。CPU 和
CUDA 调用同一份 work-item 公式。

The per-lag `synamp` uses the **same shift direction as the dot products**
(GF shifted by −lag), so the normalization obeys Cauchy–Schwarz and `cc ≤ 1`.
Both quantities are recomputed by C++ from P+S windowed obs/GF per combo — the stored
`dot_obs_gf_lag` reduction covers only depth index 1, so per-depth evaluation
cannot consume it.

Verified (2026-09-18): independent Julia reference recomputation matches OpenMP
and CUDA P+S `cc_max` to ≤1e-9 and `best_lag` exactly. Tests cover shuffled
multi-combo trials and auto/1/7/8 batch capacities.

`duration_idx` selects a Gaussian STF σ from `/paraspace/duration`. `input.jl`
precomputes each duration variant; forward includes it in the cache key and
does no runtime convolution.

## Outputs

### `status_N.h5:/intermediates/{Operator}{Phase}[_{channel}]/`

Grouped by canonical key (e.g. `XcorrP`, `XcorrS`). Deduplicated across module
instances sharing one key. Commit sequence is
`/intermediates.__tmp__` → validate/flush → backup old final → promote → flush →
remove backup. Startup validates final and recovers a valid backup when needed;
orphan temporary groups are removed. Failed preflight or compute never changes final.

| Dataset | Type | Shape (HDF5) | Description |
|------------|---------|-------------------------|--------------------------------------------------|
| `cc_max` | Float64 | `[N_phases × N_trials]` | max normalized CC per (phase, trial) |
| `best_lag` | Int32 | `[N_phases × N_trials]` | best-lag offset in samples (∈ [−maxlag, maxlag]) |

Storage is C-order `[N_phases × N_trials]`; `HDF5.jl` reads it as
`(N_trials, N_phases)` (reversed dims) — account for this in consumers.

## Known behavior notes

- `H5Lexists` probes of optional groups (`/XcorrP/...`) are silenced with
  `H5E_BEGIN_TRY`; forward produces no `HDF5-DIAG` noise on stderr.
- `station_idx` is normalized from 1-based (HDF5) to 0-based immediately after
  reading; the result is consumed only by the (deferred) Polarity path.
- Polarity/PSR kernels remain compiled but are **deferred**；当前基线只注册
  XcorrP/XcorrS。
- Active XcorrP/XcorrS modules must use identical band, trim, max-lag, dt and
  window length. Mismatch is a preflight error.

## CUDA memory and batching

CUDA allocates maximum-combo `cc/synamp/obs_norm2` buffers once, then calls
`cudaMemGetInfo`. Safety margin is `max(256 MiB, total_memory/20)`. Per trial:

```
6*sizeof(Float64) + N_phases*sizeof(Float64) + N_phases*sizeof(Int32)
```

Automatic capacity is available bytes divided by this cost, capped by total
trials and `INT_MAX/N_phases`; explicit `--cuda-batch-trials` is an additional
upper bound. MT and outputs reuse one maximum-batch allocation. Each combo uses
the default stream in strict order: reductions H2D once, then per batch MT H2D →
kernel → synchronize → outputs D2H. Complete outputs remain on host until one
transactional HDF5 commit.

CUDA launcher uses 128 threads/block. RTX 5060 Ti 构建下 kernel 使用 96
registers/thread，无 local-memory spill；64/128/256 实测接近，128 为当前默认值。

Logs include requested/selected backend, GPU name, batch capacity, CUDA
context/allocation/H2D/kernel/D2H milliseconds, evaluation time and total time.

2026-09-18 reference measurement, synthetic 455,544 trials, RTX 5060 Ti,
thread-local quadratic coefficients enabled: OpenMP 16-thread evaluation/forward
total = 3,206.6/3,738.8 ms; CUDA (128 threads/block) kernel/evaluation/forward
total = 1,705.2/1,865.8/2,474.9 ms. These are observations, not acceptance
thresholds.

## Build and verification

```bash
# CPU-only
cmake -S forward -B forward/build-cpu -DFM_ENABLE_CUDA=OFF
cmake --build forward/build-cpu -j8

# CUDA; architecture is explicit and not hardcoded in source
CUDACXX=/path/to/nvcc cmake -S forward -B forward/build-cuda \
    -DFM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build forward/build-cuda -j8

FM_FORWARD_EXE="$PWD/forward/build-cuda/forward" FM_TEST_CUDA=1 \
    julia --project=. tests/stages/forward_test.jl
./forward/build-cuda/xcorr_cuda_test
```

## What It Does NOT Do

- Does NOT write `/misfits` (assess.jl).
- Does NOT resolve trials to physical values beyond SDR→MT conversion
  (angles are resolved from `/paraspace` before the kernel).
- Does NOT apply weights or make convergence decisions.
