# Stage: `plot.sh` — Event and Station Plot

## Role

`plot.sh` renders the event and station coordinates from `database.h5` with a
selected plotting backend. The first implementation supports Gnuplot and uses
longitude as X and latitude as Y without map projection.

## Usage

```bash
bash plot.sh --data-dir <dir> --backend gnuplot
```

The default output is `<dir>/figures/event_stations.png`. Use `--output` to
choose another file. The event and station coordinates are extracted by the
data directory's Julia environment when `<dir>/Project.toml` exists.

`scripts/extract_h5.jl` converts selected scalar or one-dimensional HDF5
datasets into tab-separated columns. `scripts/gnuplot/event_stations.gp`
contains only Gnuplot rendering commands.

```bash
julia --project=<dir> scripts/extract_h5.jl \
    <input.h5> <output.dat> <dataset> [<dataset> ...]
```

Scalar columns are broadcast to the longest one-dimensional dataset. Dataset
lengths must otherwise match.

## Waveform comparison

```bash
bash plot_waveforms.sh --data-dir <dir> --backend gnuplot
```

`scripts/extract_waveform_comparison.jl` finds the best trial in the latest
status file, reconstructs synthetic waveforms from the selected Green's
functions and solution moment tensor, and applies each channel's signed
`best_lag` to the full preprocessed waveform before trimming the plotting
window. Before writing temporary Gnuplot data, it separately verifies the
stored `cc_max` using the inversion kernel's fixed-window convention.

`scripts/gnuplot/waveform_comparison.gp` renders one physical station per row
and six phase-component columns: P-Z, P-N, P-E, S-Z, S-N, and S-E. Observed
traces are black solid lines and synthetic traces are red solid lines. Axes and
grids are omitted, and P/S panel widths are proportional to their actual time
spans. The default output is
`<dir>/figures/waveform_comparison.png`.

Gnuplot must be installed separately. A headless Linux installation can use
`gnuplot-nox`.
