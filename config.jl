# Auto-generated pipeline config for synthetic test event.
# Loaded by input.jl via include() — Config module already loaded.

Config.misfit_modules() = ["XCorr", "Polarity"]
Config.module_weights() = [0.5, 0.5]
Config.minimum_stations() = 2

Config.data_file() = joinpath(@__DIR__, "raw.h5")

Config.freq_bands() = [(0.5, 2.0)]

Config.depths() = [5.0, 10.0, 15.0]

Config.grid_params() = (
    strike0 = 45.0,
    dstrike = 20.0,
    nstrike = 3,
    dip0 = 30.0,
    ddip = 20.0,
    ndip = 3,
    rake0 = 0.0,
    drake = 20.0,
    nrake = 3,
)

Config.xcorr_params() = (
    maxlag_factor = 0.5,
    filter_order = 4,
    P_trim = [-2.0, 5.0],
    S_trim = [-2.0, 5.0],
    select_threshold = 0.5,
    deselect_threshold = 0.3,
)

Config.polarity_params() = (trim = [0.0, 2.0],)

Config.greens_params() = (gf_dir = "tests/synthetic/", model = "synthetic")
