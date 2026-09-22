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

Gnuplot must be installed separately. A headless Linux installation can use
`gnuplot-nox`.
