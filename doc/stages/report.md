# Stage: `scripts/report.jl` — Markdown Report

## Role

`report.jl` is an optional presentation stage after `output.jl`. It reads the
compact `result.toml` result and writes a human-readable `report.md`. It does
not read waveforms or Green's functions and does not recompute any misfit.

## Usage

```bash
DATA_DIR=<dir> julia scripts/report.jl
# or
julia scripts/report.jl <dir>
```

The input is `<dir>/result.toml`; the output is `<dir>/report.md`. The normal
`driver.sh` workflow runs this stage automatically after `output.jl`.

## Report sections

- Summary: iteration/trial counts and convergence reason.
- Best-fit solution: SDR, depth, duration, frequency/duration indices, and
  moment tensor.
- Uncertainty: available SDR/depth statistics and frequency test curve.
- Phase quality: cross-correlation and per-module misfit by phase.
- Station summary: phase count, mean cross-correlation, and total misfit.

Missing, NaN, or not-yet-implemented values are shown as `N/A`. Misfit
columns use the module names and order recorded in `result.toml`.
