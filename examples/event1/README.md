# Event 1 example

This example loads the SAC waveforms with `SeisTools.jl` and calculates
frequency-domain Green functions with `DWN.jl`.

The configuration estimates P/S picks from the event origin and the supplied
1-D model. These are initial picks for a pipeline smoke test and should be
replaced with analyst-reviewed picks before interpreting the inversion result.

Run from the repository root:

```bash
EVENT1_GF_NPTS=8192 EVENT1_DWN_MAX_ORDER=100 \
bash driver.sh --data-dir examples/event1 --trial-budget 10000
```

`driver.sh` detects `examples/event1/Project.toml` and uses this project
environment automatically. Install its dependencies once with:

```bash
julia --project=examples/event1 -e 'using Pkg; Pkg.instantiate()'
```

The installation uses Julia's normal depot and does not create a separate
temporary depot.

`EVENT1_GF_NPTS` and `EVENT1_DWN_MAX_ORDER` are optional controls for the DWN
calculation. Larger values improve the Green-function calculation but increase
runtime and memory use. The generated database, status, and report files stay
in this directory and are ignored by Git.

To remove generated pipeline outputs while preserving the SAC inputs and event
configuration, run:

```bash
bash examples/event1/clean.sh
```

To plot the event and station distribution directly in longitude/latitude
coordinates (without map projection), run:

```bash
bash plot.sh --data-dir examples/event1 --backend gnuplot
```

The default output is `examples/event1/figures/event_stations.png`. Use
`--output <file>` to choose another output path.

To compare observed and best-trial synthetic waveforms after cross-correlation
alignment, run:

```bash
bash plot_waveforms.sh --data-dir examples/event1 --backend gnuplot
```

The waveform figure uses one row per station and six phase-component columns:
P-Z, P-N, P-E, S-Z, S-N, and S-E. Observed traces are black and synthetic
traces are red solid lines. Synthetic traces are aligned on the full
preprocessed waveform before the plotting window is trimmed. Axes and grids
are omitted. P/S panel widths are proportional to their actual time spans. The default output is
`examples/event1/figures/waveform_comparison.png`.
