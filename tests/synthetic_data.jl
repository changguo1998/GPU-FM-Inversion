#!/usr/bin/env julia
#
# synthetic_data.jl — Generate minimal raw.h5 and config.jl for pipeline testing.
#
# Usage:
#   julia tests/synthetic_data.jl                          # writes to CWD
#   julia tests/synthetic_data.jl /tmp/test_event           # writes to /tmp/test_event/
#
# Deterministic: uses Random.seed!(42). Overwrites existing files.

using HDF5
using Random
using Dates

# ── Arguments ──────────────────────────────────────────────────────────────────
outdir = length(ARGS) > 0 ? ARGS[1] : "."
mkpath(outdir)
raw_h5 = joinpath(outdir, "raw.h5")
cfg_jl = joinpath(outdir, "config.jl")

# ── Deterministic RNG ─────────────────────────────────────────────────────────
Random.seed!(42)

# ── 1. /event group ───────────────────────────────────────────────────────────
event = Dict(
    "longitude"  => 120.0,
    "latitude"   => 30.0,
    "depth"      => 10.0,
    "magnitude"  => 5.0,
    "origintime" => "2024-01-01T00:00:00",
)

# ── 2. /phase_picks group ────────────────────────────────────────────────────
station_ids = ["NET.ST1", "NET.ST2", "NET.ST3"]
P_time      = ["2024-01-01T00:00:10", "2024-01-01T00:00:12", "2024-01-01T00:00:14"]
S_time      = ["2024-01-01T00:00:18", "2024-01-01T00:00:21", "2024-01-01T00:00:24"]
P_polarity  = Int8[1, -1, 0]

# ── 3. /stations group ───────────────────────────────────────────────────────
# Phase key: NET.STn.Z.{P|S}
station_entries = [
    ("NET.ST1.Z.P", 30.5, 120.5, 500.0),
    ("NET.ST1.Z.S", 30.5, 120.5, 500.0),
    ("NET.ST2.Z.P", 29.5, 119.5, 600.0),
    ("NET.ST2.Z.S", 29.5, 119.5, 600.0),
    ("NET.ST3.Z.P", 30.0, 120.0, 550.0),
    ("NET.ST3.Z.S", 30.0, 120.0, 550.0),
]

n_phases = length(station_entries)
ids      = [e[1] for e in station_entries]
nets     = ["NET" for _ in 1:n_phases]
stas     = ["ST1", "ST1", "ST2", "ST2", "ST3", "ST3"]
comps    = ["Z"  for _ in 1:n_phases]
lats     = [e[2] for e in station_entries]
lons     = [e[3] for e in station_entries]
elevs    = [e[4] for e in station_entries]
dts      = fill(0.01, n_phases)                # 100 Hz
begin_t  = ["2024-01-01T00:00:05" for _ in 1:n_phases]

# ── 4. /waveforms group ──────────────────────────────────────────────────────
n_samples = 2000
waveforms = Dict{String, Vector{Float64}}()
for id in ids
    waveforms[id] = randn(Float64, n_samples)   # seeded above
end

# ── Write raw.h5 ──────────────────────────────────────────────────────────────
h5open(raw_h5, "w") do file
    # /event
    g_event = create_group(file, "/event")
    g_event["longitude"]  = event["longitude"]
    g_event["latitude"]   = event["latitude"]
    g_event["depth"]      = event["depth"]
    g_event["magnitude"]  = event["magnitude"]
    g_event["origintime"] = event["origintime"]

    # /phase_picks
    g_phases = create_group(file, "/phase_picks")
    g_phases["station_ids"]  = station_ids
    g_phases["P_time"]       = P_time
    g_phases["S_time"]       = S_time
    g_phases["P_polarity"]   = P_polarity

    # /stations
    g_stations = create_group(file, "/stations")
    g_stations["id"]         = ids
    g_stations["network"]    = nets
    g_stations["station"]    = stas
    g_stations["component"]  = comps
    g_stations["latitude"]   = lats
    g_stations["longitude"]  = lons
    g_stations["elevation"]  = elevs
    g_stations["dt"]         = dts
    g_stations["begin_time"] = begin_t

    # /waveforms  (one dataset per phase id)
    g_wave = create_group(file, "/waveforms")
    for (id, data) in waveforms
        g_wave[id] = data
    end
end

# ── Write config.jl ───────────────────────────────────────────────────────────
config = """\
# Auto-generated pipeline config for synthetic test event.
# Loaded by input.jl via include() — Config module already loaded.

Config.misfit_modules()   = ["XCorr", "Polarity", "PSR"]
Config.module_weights()   = [0.5, 0.25, 0.25]
Config.minimum_stations() = 2

Config.data_file() = joinpath(@__DIR__, "raw.h5")

Config.freq_bands() = [(0.5, 2.0)]

Config.depths() = [5.0, 10.0, 15.0]

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

Config.xcorr_params() = (
    maxlag_factor      = 0.5,
    filter_order       = 4,
    P_trim             = [-2.0, 5.0],
    S_trim             = [-2.0, 5.0],
    select_threshold   = 0.5,
    deselect_threshold = 0.3,
)

Config.polarity_params() = (trim = [0.0, 2.0],)

Config.greens_params() = (gf_dir = "tests/synthetic/", model = "synthetic",)

Config.freq_test_max_iter() = 3
"""

write(cfg_jl, config)

# ── Summary ───────────────────────────────────────────────────────────────────
println("Synthetic test data generated in: $(realpath(outdir))")
println("  data file  — raw.h5 (/event, /phase_picks, /stations, /waveforms)")
println("  config.jl  — pipeline config (3 stations, 1 freq band, 3 depths, 3x3x3 grid)")
println("  stations  : $(join(station_ids, ", "))")
println("  phases    : $(length(ids)) phase-station pairs")
println("  waveform  : $(n_samples) samples per phase (Float64, RNG seed=42)")