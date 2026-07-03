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

# Arguments

outdir = length(ARGS) > 0 ? ARGS[1] : "."
mkpath(outdir)
raw_h5 = joinpath(outdir, "raw.h5")
cfg_jl = joinpath(outdir, "config.jl")

# Deterministic RNG

Random.seed!(42)

# Constants

n_stations = 3
n_samples = 2000
dt = 0.01

# 1. /event group

event = Dict(
    "longitude" => 120.0,
    "latitude" => 30.0,
    "depth" => 10.0,
    "magnitude" => 5.0,
    "origintime" => "2024-01-01T00:00:00",
)

# 2. /phase_picks group (one per station)

station_ids = ["NET.ST1", "NET.ST2", "NET.ST3"]
P_time = ["2024-01-01T00:00:10", "2024-01-01T00:00:12", "2024-01-01T00:00:14"]
S_time = ["2024-01-01T00:00:18", "2024-01-01T00:00:21", "2024-01-01T00:00:24"]
P_polarity = Int8[1, -1, 0]

# 3. /stations group (one per station)

sta_ids = ["NET.ST1", "NET.ST2", "NET.ST3"]
nets = ["NET", "NET", "NET"]
stas = ["ST1", "ST2", "ST3"]
chans = ["Z", "Z", "Z"]
lats = [30.5, 29.5, 30.0]
lons = [120.5, 119.5, 120.0]
elevs = [500.0, 600.0, 550.0]
dts = fill(dt, n_stations)
begin_t = ["2024-01-01T00:00:05" for _ in 1:n_stations]

# 4. /waveforms group (one per channel, key = "{station_id}.{channel}")

channel_ids = [sta_id * "." * ch for (sta_id, ch) in zip(stas, chans)]
# With network prefix for full channel_id: NET.ST1.Z
full_channel_ids = [nets[i] * "." * channel_ids[i] for i in 1:n_stations]

waveforms = Dict{String, Vector{Float64}}()
for ch_id in full_channel_ids
    waveforms[ch_id] = randn(Float64, n_samples)
end

# Write raw.h5

h5open(raw_h5, "w") do file
    # /event
    g_event = create_group(file, "/event")
    g_event["longitude"] = event["longitude"]
    g_event["latitude"] = event["latitude"]
    g_event["depth"] = event["depth"]
    g_event["magnitude"] = event["magnitude"]
    g_event["origintime"] = event["origintime"]

    # /phase_picks
    g_phases = create_group(file, "/phase_picks")
    g_phases["station_ids"] = station_ids
    g_phases["P_time"] = P_time
    g_phases["S_time"] = S_time
    g_phases["P_polarity"] = P_polarity

    # /stations — one row per station
    g_stations = create_group(file, "/stations")
    g_stations["id"] = sta_ids
    g_stations["network"] = nets
    g_stations["station"] = stas
    g_stations["channel"] = chans
    g_stations["latitude"] = lats
    g_stations["longitude"] = lons
    g_stations["elevation"] = elevs
    g_stations["dt"] = dts
    g_stations["begin_time"] = begin_t

    # /waveforms — one dataset per channel_id
    g_wave = create_group(file, "/waveforms")
    for (ch_id, data) in waveforms
        g_wave[ch_id] = data
    end
end

# Helper: extract channel_id from phase_id (drop phase type suffix)
# Phase key: NET.ST1.Z.P  →  channel_id: NET.ST1.Z
_channel_from_phase(pid) = join(split(pid, ".")[1:3], ".")

# Write config.jl

config = """\
# Auto-generated pipeline config for synthetic test event.
# Loaded by input.jl via include() — Config module already loaded.

Config.misfit_modules()   = ["XCorr", "Polarity"]
Config.minimum_stations() = 2

Config.freq_bands() = [(0.5, 2.0)]

Config.depths() = [5.0, 10.0, 15.0]

Config.xcorr_params() = (
    maxlag_factor      = 0.5,
    filter_order       = 4,
    P_trim             = [-2.0, 5.0],
    S_trim             = [-2.0, 5.0],
    select_threshold   = 0.5,
    deselect_threshold = 0.3,
)

Config.polarity_params() = (trim = [0.0, 2.0],)

# Data reading — reads from raw.h5 (test file generated alongside)

const _RAW_H5 = joinpath(@__DIR__, "raw.h5")

Config.load_event()       = IO.read_event(_RAW_H5)
Config.load_stations()    = IO.read_stations(_RAW_H5)
Config.load_phase_picks() = IO.read_phase_picks(_RAW_H5)

Config.load_waveform(pid::String) = begin
    # Extract channel_id from phase_id (drop phase type suffix)
    # "NET.ST1.Z.P" → "NET.ST1.Z"
    parts = split(pid, ".")
    ch_id = join(parts[1:3], ".")
    IO.read_waveform(_RAW_H5, ch_id)
end

using Random

Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
    nt = 2000
    dt = 0.01
    tp = 5.0
    ts = 13.0
    rng = Random.MersenneTwister(42 + round(Int, src_depth) + round(Int, sta_lat * 10))
    gf_raw = randn(rng, nt, 6, 3)
    decay = exp.(-(0:(nt - 1)) ./ (nt / 4))
    for c in 1:3, m in 1:6
        gf_raw[:, m, c] .*= decay
    end
    return (gf_raw, dt, tp, ts)
end
"""

write(cfg_jl, config)

# Summary

println("Synthetic test data generated in: $(realpath(outdir))")
println("  data file  — raw.h5 (/event, /phase_picks, /stations, /waveforms)")
println("  config.jl  — pipeline config (3 stations, 1 freq band, 3 depths, 3x3x3 grid)")
println("  stations  : $(join(sta_ids, ", "))")
println("  channels  : $(join(full_channel_ids, ", "))")
println("  waveform  : $(n_samples) samples per channel (Float64, RNG seed=42)")
