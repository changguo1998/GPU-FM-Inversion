#!/usr/bin/env julia
#
# synthetic_data.jl — Generate test raw.h5 and config.jl with realistic synthetic data.
#
# Generates raw.h5 (/event, /phase_picks, /stations, /waveforms) and config.jl
# for a synthetic event.  Observed waveforms are consistent with Green's functions:
# obs = GF * MT + noise, where MT is derived from a known SDR.
#
# Usage:
#   julia tests/synthetic_data.jl                          # writes to CWD
#   julia tests/synthetic_data.jl /tmp/test_event           # writes to /tmp/test_event/
#   julia tests/synthetic_data.jl --nsta 5 --npts 4000
#   julia tests/synthetic_data.jl /tmp/test_event --nsta 5
#
# Deterministic: uses Random.seed!(42). Overwrites existing files.

using HDF5
using Random
using Dates
using IO

# ---------------------------------------------------------------------------
# Key constants (overridable via CLI --key value)
# ---------------------------------------------------------------------------

const DEFAULT_N_STATION = 3
const DEFAULT_NPTS = 2000
const DEFAULT_DT = 0.01

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

_outdir = "."
_n_station = DEFAULT_N_STATION
_npts = DEFAULT_NPTS
_dt = DEFAULT_DT

let
    local i = 1
    local od = _outdir
    local ns = _n_station
    local np = _npts
    local d = _dt
    while i <= length(ARGS)
        if ARGS[i] == "--nsta"
            ns = parse(Int, ARGS[i + 1]);
            i += 2
        elseif ARGS[i] == "--npts"
            np = parse(Int, ARGS[i + 1]);
            i += 2
        elseif ARGS[i] == "--dt"
            d = parse(Float64, ARGS[i + 1]);
            i += 2
        elseif startswith(ARGS[i], "--")
            error("Unknown flag: $(ARGS[i])")
        else
            od = ARGS[i];
            i += 1
        end
    end
    global _outdir = od
    global _n_station = ns
    global _npts = np
    global _dt = d
end

outdir = _outdir
n_station = _n_station
npts = _npts
dt = _dt

mkpath(outdir)
raw_h5 = joinpath(outdir, "raw.h5")
cfg_jl = joinpath(outdir, "config.jl")

# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

Random.seed!(42)

# ---------------------------------------------------------------------------
# 1. Source parameters
# ---------------------------------------------------------------------------

strike = 30.0
dip = 60.0
rake = 90.0

using MT
mt_true = MT.sdr_to_mt(strike, dip, rake)

# ---------------------------------------------------------------------------
# 2. Station geometry
# ---------------------------------------------------------------------------

# Place stations at increasing distances from epicenter, various azimuths
azimuths = range(0.0, 315.0, length = n_station)
dists_km = range(10.0, 80.0, length = n_station)

sta_ids = String[]
nets = String[]
stas = String[]
chans = String[]
lats = Float64[]
lons = Float64[]
elevs = Float64[]

event_lat = 30.0
event_lon = 120.0
event_depth = 10.0

for i in 1:n_station
    az_rad = deg2rad(azimuths[i])
    dist_deg = dists_km[i] / 111.0
    lat = event_lat + dist_deg * cos(az_rad)
    lon = event_lon + dist_deg * sin(az_rad)
    push!(sta_ids, "NET.ST$i")
    push!(nets, "NET")
    push!(stas, "ST$i")
    push!(chans, "Z")
    push!(lats, round(lat, digits = 4))
    push!(lons, round(lon, digits = 4))
    push!(elevs, 500.0 + i * 50.0)
end

begin_t = ["2024-01-01T00:00:05" for _ in 1:n_station]

# Distance and azimuth (degrees from event to each station)
dists = Float64[]
azims = Float64[]
for i in 1:n_station
    d_km = IO.haversine_distance(event_lat, event_lon, lats[i], lons[i])
    az = IO.compute_azimuth(event_lat, event_lon, lats[i], lons[i])
    push!(dists, round(d_km, digits = 1))
    push!(azims, round(az, digits = 1))
end

# ---------------------------------------------------------------------------
# 3. Travel times (simple 1D velocity model, straight-ray)
# ---------------------------------------------------------------------------

vp = 6.0
vs = 3.5

P_times = String[]
S_times = String[]
origin_dt = DateTime(2024, 1, 1, 0, 0, 0)

for i in 1:n_station
    dist_m = dists[i] * 1000.0
    hypocentral = sqrt(dist_m^2 + (event_depth * 1000.0)^2)
    tp = hypocentral / (vp * 1000.0)
    ts = hypocentral / (vs * 1000.0)
    p_time = origin_dt + Millisecond(round(Int, tp * 1000))
    s_time = origin_dt + Millisecond(round(Int, ts * 1000))
    push!(P_times, Dates.format(p_time, "yyyy-mm-ddTHH:MM:SS"))
    push!(S_times, Dates.format(s_time, "yyyy-mm-ddTHH:MM:SS"))
end

# ---------------------------------------------------------------------------
# 4. Polarity from focal mechanism (simplified radiation pattern)
# ---------------------------------------------------------------------------

P_polarity = Int8[]
for i in 1:n_station
    az_rad = deg2rad(azimuths[i])
    takeoff = atan(dists[i] * 1000.0, event_depth * 1000.0)
    # Simplified P-wave radiation pattern for dip-slip
    amp = sin(2 * az_rad) * sin(takeoff)^2
    if amp > 0.1
        push!(P_polarity, Int8(1))
    elseif amp < -0.1
        push!(P_polarity, Int8(-1))
    else
        push!(P_polarity, Int8(0))
    end
end

# ---------------------------------------------------------------------------
# 5. Green's functions per (station, depth)
# ---------------------------------------------------------------------------

depths = [5.0, 10.0, 15.0]

# GF shape: [nt, 6, 3], channel order [N, E, D]
# Seed MUST match Config.load_gf so obs == GF * MT + noise is consistent.

function _gf_seed(src_depth, sta_lat, sta_lon)::Int
    round(Int, src_depth) + round(Int, sta_lat * 10) + round(Int, sta_lon * 10)
end

function _generate_gf(seed::Int, nt::Int)::Array{Float64, 3}
    rng = Random.MersenneTwister(seed)
    gf = randn(rng, Float64, nt, 6, 3)
    decay = exp.(-(0:(nt - 1)) ./ (nt / 4))
    for c in 1:3, m in 1:6
        gf[:, m, c] .*= decay
    end
    return gf
end

gf_dict = Dict{Tuple{Int, Int}, Array{Float64, 3}}()
for (di, depth) in enumerate(depths)
    for si in 1:n_station
        seed = _gf_seed(depth, lats[si], lons[si])
        gf_dict[(di, si)] = _generate_gf(seed, npts)
    end
end

# ---------------------------------------------------------------------------
# 6. Synthetic observed waveforms: obs = GF * MT + noise
# ---------------------------------------------------------------------------

# Z component only (channel index 3 = D/vertical)
ch_index = 3

waveforms = Dict{String, Vector{Float64}}()
noise_rng = Random.MersenneTwister(999)

for si in 1:n_station
    ch_id = sta_ids[si] * ".Z"

    # Use shallowest depth GF to generate obs
    gf = gf_dict[(1, si)]
    syn = zeros(Float64, npts)
    for m in 1:6
        syn .+= gf[:, m, ch_index] .* mt_true[m]
    end

    # 10% noise
    rms = sqrt(sum(syn .^ 2) / npts)
    noise = randn(noise_rng, Float64, npts) .* (rms * 0.1)
    waveforms[ch_id] = syn .+ noise
end

# ---------------------------------------------------------------------------
# 7. Write raw.h5
# ---------------------------------------------------------------------------

h5open(raw_h5, "w") do file
    # /event
    g_event = create_group(file, "event")
    g_event["longitude"] = event_lon
    g_event["latitude"] = event_lat
    g_event["depth"] = event_depth
    g_event["magnitude"] = 5.0
    g_event["origintime"] = "2024-01-01T00:00:00"

    # /phase_picks
    g_phases = create_group(file, "phase_picks")
    g_phases["station_ids"] = sta_ids
    g_phases["P_time"] = P_times
    g_phases["S_time"] = S_times
    g_phases["P_polarity"] = P_polarity

    # /stations
    g_stations = create_group(file, "stations")
    g_stations["id"] = sta_ids
    g_stations["network"] = nets
    g_stations["station"] = stas
    g_stations["channel"] = chans
    g_stations["latitude"] = lats
    g_stations["longitude"] = lons
    g_stations["elevation"] = elevs
    g_stations["dt"] = fill(dt, n_station)
    g_stations["begin_time"] = begin_t

    # /waveforms
    g_wave = create_group(file, "waveforms")
    for (ch_id, data) in waveforms
        g_wave[ch_id] = data
    end
end

# ---------------------------------------------------------------------------
# 8. Helper: format depths list for config string embedding
# ---------------------------------------------------------------------------

_join_depths(d) = join([_fmt_depth(v) for v in d], ", ")
_fmt_depth(v::Float64) = string(v)
_fmt_depth(v::Int) = string(v)

# ---------------------------------------------------------------------------
# 9. Write config.jl
# ---------------------------------------------------------------------------

# Embed GF generation functions so Config.load_gf() produces the exact same
# GF as used to generate synthetic waveforms.

config = """\
# Auto-generated pipeline config for synthetic test event.
# Loaded by input.jl via include() — Config module already loaded.

Config.misfit_modules()   = ["XCorr", "Polarity"]
Config.minimum_stations() = 2

Config.freq_bands() = [(0.5, 2.0)]

Config.depths() = [$(_join_depths(depths))]

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
    parts = split(pid, ".")
    ch_id = join(parts[1:3], ".")
    IO.read_waveform(_RAW_H5, ch_id)
end

using Random

# GF generation — must match synthetic_data.jl
function _gf_seed(src_depth, sta_lat, sta_lon)
    round(Int, src_depth) + round(Int, sta_lat * 10) + round(Int, sta_lon * 10)
end

function _generate_gf(seed, nt)
    rng = Random.MersenneTwister(seed)
    gf = randn(rng, Float64, nt, 6, 3)
    decay = exp.(-(0:(nt - 1)) ./ (nt / 4))
    for c in 1:3, m in 1:6
        gf[:, m, c] .*= decay
    end
    return gf
end

Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
    nt = $npts
    _dt = $dt
    vp = 6.0
    vs = 3.5
    dist_m = IO.haversine_distance(src_lat, src_lon, sta_lat, sta_lon) * 1000.0
    hypocentral = sqrt(dist_m^2 + (src_depth * 1000.0)^2)
    tp = hypocentral / (vp * 1000.0)
    ts = hypocentral / (vs * 1000.0)
    seed = _gf_seed(src_depth, sta_lat, sta_lon)
    gf_raw = _generate_gf(seed, nt)
    return (gf_raw, _dt, tp, ts)
end
"""

write(cfg_jl, config)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

println("Synthetic test data generated in: $(realpath(outdir))")
println("  data file  — raw.h5 (/event, /phase_picks, /stations, /waveforms)")
println(
    "  config.jl  — pipeline config ($n_station stations, 1 freq band, $(length(depths)) depths)",
)
println("  SDR        — strike=$(strike), dip=$(dip), rake=$(rake)")
println("  MT         — [$(join(round.(mt_true, digits=4), ", "))]")
println("  stations   : $(join(sta_ids, ", "))")
println("  distances  : $(join(dists, ", ")) km")
println("  P times    : $(join(P_times, ", "))")
println("  S times    : $(join(S_times, ", "))")
println("  waveform   : $npts samples @ $(dt) s (obs = GF * MT + noise)")
