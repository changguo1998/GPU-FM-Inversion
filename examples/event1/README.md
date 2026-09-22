# Event 1 example

This example loads the SAC waveforms with `SeisTools.jl` and calculates
frequency-domain Green functions with `DWN.jl`.

The configuration estimates P/S picks from the event origin and the supplied
1-D model. These are initial picks for a pipeline smoke test and should be
replaced with analyst-reviewed picks before interpreting the inversion result.

Run from the repository root:

```bash
JULIA_DEPOT_PATH=/tmp/refactor-fm-julia-depot:$HOME/.julia \
EVENT1_GF_NPTS=8192 EVENT1_DWN_MAX_ORDER=100 \
bash driver.sh --data-dir examples/event1 --trial-budget 10000
```

`EVENT1_GF_NPTS` and `EVENT1_DWN_MAX_ORDER` are optional controls for the DWN
calculation. Larger values improve the Green-function calculation but increase
runtime and memory use. The generated database, status, and report files stay
in this directory and are ignored by Git.
