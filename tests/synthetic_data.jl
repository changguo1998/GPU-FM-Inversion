#!/usr/bin/env julia
#
# synthetic_data.jl — Generate synthetic test data as plain files.
#
# Outputs:
#   1. stations.txt  — station list (table)
#   2. {sta}.{ch}.dat — waveform per station+channel (one column)
#   3. phases.txt   — phase picks table (station, P_time, S_time)
#
# Usage:
#   julia tests/synthetic_data.jl                          # writes to CWD
#   julia tests/synthetic_data.jl /tmp/test_event
#   julia tests/synthetic_data.jl --nsta 5 --npts 4000
#
# Deterministic: Random.seed!(42). Overwrites existing files.

using Random
using Dates

# ---------------------------------------------------------------------------
# Key constants (overridable via CLI --key value)
# ---------------------------------------------------------------------------

const DEFAULT_N_STATION = 6
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

# Simplified MT in NED: [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
# 30/60/90 dip-slip → roughly [-0.2165, -0.6495, 0.866, 0.375, 0.25, -0.433]
mt_true = [-0.2165, -0.6495, 0.866, 0.375, 0.25, -0.433]

# ---------------------------------------------------------------------------
# 2. Station geometry
# ---------------------------------------------------------------------------

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

begin_t = "2024-01-01T00:00:05"

# Distance from event (km) — simple flat-earth approximation
dists = Float64[]
for i in 1:n_station
    dlon = deg2rad(lons[i] - event_lon)
    dlat = deg2rad(lats[i] - event_lat)
    a = sin(dlat / 2)^2 + cos(deg2rad(event_lat)) * cos(deg2rad(lats[i])) * sin(dlon / 2)^2
    d = 2 * 6371.0 * asin(sqrt(a))
    push!(dists, round(d, digits = 1))
end

# ---------------------------------------------------------------------------
# 3. Travel times (1D velocity model, straight-ray)
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
# 4. Green's functions per (station, depth)
# ---------------------------------------------------------------------------

depths = [5.0, 10.0, 15.0]

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
# 5. Synthetic observed waveforms: obs = GF * MT + noise
# ---------------------------------------------------------------------------

ch_index = 3  # D/Z vertical

waveforms = Dict{String, Vector{Float64}}()
noise_rng = Random.MersenneTwister(999)

for si in 1:n_station
    ch_id = sta_ids[si] * ".Z"
    gf = gf_dict[(1, si)]
    syn = zeros(Float64, npts)
    for m in 1:6
        syn .+= gf[:, m, ch_index] .* mt_true[m]
    end
    rms = sqrt(sum(syn .^ 2) / npts)
    noise = randn(noise_rng, Float64, npts) .* (rms * 0.1)
    waveforms[ch_id] = syn .+ noise
end

# ---------------------------------------------------------------------------
# 6. Write output files
# ---------------------------------------------------------------------------

# 6a. Station list file
open(joinpath(outdir, "stations.txt"), "w") do f
    write(
        f,
        "# station_id  network  station  channel  latitude  longitude  elevation  dt  begin_time\n",
    )
    for i in 1:n_station
        write(
            f,
            "$(sta_ids[i])  $(nets[i])  $(stas[i])  $(chans[i])  $(lats[i])  $(lons[i])  $(elevs[i])  $dt  $begin_t\n",
        )
    end
end

# 6b. Waveform files — one per station+channel
for (ch_id, data) in waveforms
    fn = ch_id * ".dat"
    open(joinpath(outdir, fn), "w") do f
        for v in data
            write(f, "$v\n")
        end
    end
end

# 6c. Phase picks file
open(joinpath(outdir, "phases.txt"), "w") do f
    write(f, "# station_id  P_time  S_time\n")
    for i in 1:n_station
        write(f, "$(sta_ids[i])  $(P_times[i])  $(S_times[i])\n")
    end
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

println("Synthetic test data generated in: $(realpath(outdir))")
println("  stations.txt  — station list ($n_station stations)")
println("  phases.txt    — phase picks table")
for (ch_id, _) in waveforms
    println("  $(ch_id).dat  — waveform ($npts samples)")
end
println("  SDR           — strike=$(strike), dip=$(dip), rake=$(rake)")
println("  distances     : $(join(dists, ", ")) km")
