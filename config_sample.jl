#
# config_sample.jl — Sample pipeline configuration
#
# Copy this file and edit the return values for your dataset.
# Each function MUST be implemented — if one is missing, you'll
# get a ConfigError at startup telling you exactly what to define.
#
# The Config module is loaded by input.jl before this file is included.
# When running standalone for validation, uncomment the include/using block.
#
# Usage:
#   julia scripts/input.jl data/raw.h5 config_sample.jl

# (Uncomment the two lines below only for standalone validation)
# include("shared/config/src/Config.jl")
# using .Config

# ── Misfit modules ────────────────────────────────────────
Config.misfit_modules()   = ["XCorr", "Polarity", "PSR"]
Config.module_weights()   = [0.5, 0.25, 0.25]
Config.minimum_stations() = 2

# ── Frequency bands ──────────────────────────────────────
Config.freq_bands() = [(0.5, 2.0)]

# ── Depth range ──────────────────────────────────────────
Config.depths() = [5.0, 10.0, 15.0]

# ── Initial grid ─────────────────────────────────────────
Config.grid_params() = (
    strike0 = 45.0,
    dstrike = 20.0,
    nstrike = 3,
    dip0    = 30.0,
    ddip    = 20.0,
    ndip    = 3,
    rake0   = 0.0,
    drake   = 20.0,
    nrake   = 3,
)

# ── XCorr module ─────────────────────────────────────────
Config.xcorr_params() = (
    maxlag_factor      = 0.5,
    filter_order       = 4,
    P_trim             = [-2.0, 5.0],
    S_trim             = [-2.0, 5.0],
    select_threshold   = 0.5,
    deselect_threshold = 0.3,
)

# ── Polarity module ──────────────────────────────────────
Config.polarity_params() = (trim = [0.0, 2.0],)

# ── Green's functions ────────────────────────────────────
Config.greens_params() = (gf_dir = "tests/synthetic/", model = "synthetic",)

# ── Frequency test ───────────────────────────────────────
Config.freq_test_max_iter() = 3