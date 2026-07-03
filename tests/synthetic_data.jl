#!/usr/bin/env julia
#
# synthetic_data.jl — Generate synthetic test data as plain files.
#
# Physics: far-field P/S waves in two-layer half-space.
#   Upper layer (0–20 km): vp=6, vs=4 km/s
#   Lower layer (20 km+):  vp=8, vs=6 km/s
# GF = delta impulses at direct + reflected (interface at 20 km) arrivals,
# amplitude scaled by A_const / r² / v³ with full moment-tensor radiation
# pattern. Reflection coefficients from normal-incidence impedance contrast.
# Observed = GF * MT + noise.
#
# Source at origin (lat=0, lon=0, depth=10 km).
# Stations randomly placed around source.
#
# Outputs:
#   stations.txt  — station_id  lat  lon
#   {net}.{sta}.{ch}.dat — waveform (one column)
#   phases.txt    — station_id  P_time  S_time
#
# Usage:
#   julia tests/synthetic_data.jl                    # writes to CWD
#   julia tests/synthetic_data.jl /tmp/test_event
#   julia tests/synthetic_data.jl --strike 30 --dip 60 --rake 90 --nsta 6
#
# Deterministic: Random.seed!(42). Overwrites existing files.

using Random
using Dates

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------

const DEFAULT_N_STATION = 12
const DEFAULT_NPTS = 2000
const DEFAULT_DT = 0.01
const DEFAULT_STRIKE = 30.0
const DEFAULT_DIP = 60.0
const DEFAULT_RAKE = 90.0
const DEFAULT_EVENT_DEPTH = 10.0  # km
const AMPLITUDE_SCALE = 1.0e6    # A_const

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

_outdir = "."
_n_station = DEFAULT_N_STATION
_npts = DEFAULT_NPTS
_dt = DEFAULT_DT
_strike = DEFAULT_STRIKE
_dip = DEFAULT_DIP
_rake = DEFAULT_RAKE

let
    local i = 1
    local od = _outdir
    local ns = _n_station
    local np = _npts
    local d = _dt
    local sk = _strike
    local dp = _dip
    local rk = _rake
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
        elseif ARGS[i] == "--strike"
            sk = parse(Float64, ARGS[i + 1]);
            i += 2
        elseif ARGS[i] == "--dip"
            dp = parse(Float64, ARGS[i + 1]);
            i += 2
        elseif ARGS[i] == "--rake"
            rk = parse(Float64, ARGS[i + 1]);
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
    global _strike = sk
    global _dip = dp
    global _rake = rk
end

outdir = _outdir
n_station = _n_station
npts = _npts
dt = _dt
strike = _strike
dip = _dip
rake = _rake
event_depth = DEFAULT_EVENT_DEPTH

mkpath(outdir)

# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

Random.seed!(42)

# ---------------------------------------------------------------------------
# 1. Source parameters: SDR → MT
# ---------------------------------------------------------------------------

function sdr_to_mt(s, d, r)
    sd = sind(d)
    cd = cosd(d)
    ss = sind(s)
    cs = cosd(s)
    sr = sind(r)
    cr = cosd(r)
    Mxx = -(sd * cr * sind(2s) + sin(2d) * sr * ss^2)
    Myy = sd * cr * sind(2s) - sin(2d) * sr * cs^2
    Mzz = sin(2d) * sr
    Mxy = sd * cr * cosd(2s) + 0.5 * sin(2d) * sr * sind(2s)
    Mxz = -(cd * cr * cs + cosd(2d) * sr * ss)
    Myz = -(cd * cr * ss - cosd(2d) * sr * cs)
    return [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
end

mt_true = sdr_to_mt(strike, dip, rake)
norm_mt = sqrt(sum(mt_true .^ 2))
if norm_mt > 1e-12
    mt_true ./= norm_mt
end

# ---------------------------------------------------------------------------
# 2. Two-layer velocity model
# ---------------------------------------------------------------------------

const INTERFACE_DEPTH = 20.0  # km
const VP_UPPER = 6.0   # km/s
const VS_UPPER = 4.0   # km/s
const VP_LOWER = 8.0   # km/s
const VS_LOWER = 6.0   # km/s

# Normal-incidence reflection coefficients (assume equal density)
const R_PP = (VP_LOWER - VP_UPPER) / (VP_LOWER + VP_UPPER)  # ≈ 0.143
const R_SS = (VS_LOWER - VS_UPPER) / (VS_LOWER + VS_UPPER)  # ≈ 0.2

# ---------------------------------------------------------------------------
# 3. Station geometry — random around origin
# ---------------------------------------------------------------------------

sta_ids = String[]
lats = Float64[]
lons = Float64[]

dist_km_vals = 10.0 .+ rand(n_station) .* 90.0
az_deg_vals = rand(n_station) .* 360.0

for i in 1:n_station
    az_rad = deg2rad(az_deg_vals[i])
    dist_deg = dist_km_vals[i] / 111.0
    lat = dist_deg * cos(az_rad)
    lon = dist_deg * sin(az_rad)
    push!(sta_ids, "NET.ST$i")
    push!(lats, round(lat, digits = 5))
    push!(lons, round(lon, digits = 5))
end

# ---------------------------------------------------------------------------
# 4. Travel times + ray geometry
# ---------------------------------------------------------------------------

origin_dt = DateTime(2024, 1, 1, 0, 0, 0)

P_times = String[]
S_times = String[]

r_km = Float64[]         # horizontal distance (km)
tp_dir_sec = Float64[]   # direct P travel time
ts_dir_sec = Float64[]   # direct S travel time
tp_ref_sec = Float64[]   # reflected P travel time
ts_ref_sec = Float64[]   # reflected S travel time

# Direction cosines for direct wave (source → station)
γd_E = Float64[]
γd_N = Float64[]
γd_D = Float64[]

# Direction cosines for reflected wave (source → interface, downward leg)
# Using image source at depth (2*INTERFACE_DEPTH - event_depth) = 30 km
γr_E = Float64[]
γr_N = Float64[]
γr_D = Float64[]

for i in 1:n_station
    # Horizontal distance (great-circle approximation)
    dlon = deg2rad(lons[i])
    dlat = deg2rad(lats[i])
    a = sin(dlat / 2)^2 + cos(deg2rad(lats[i])) * sin(dlon / 2)^2
    d_km = 2 * 6371.0 * asin(sqrt(a))
    d_km = max(d_km, 0.001)
    push!(r_km, d_km)

    # Direct wave: hypocentral distance
    rh_dir = sqrt(d_km^2 + event_depth^2)
    push!(tp_dir_sec, rh_dir / VP_UPPER)
    push!(ts_dir_sec, rh_dir / VS_UPPER)

    # Direct wave direction cosines
    ve = lons[i] * 111.0 * 1000.0
    vn = lats[i] * 111.0 * 1000.0
    vd = -event_depth * 1000.0
    vnorm = sqrt(ve^2 + vn^2 + vd^2)
    if vnorm > 0
        push!(γd_E, ve / vnorm)
        push!(γd_N, vn / vnorm)
        push!(γd_D, vd / vnorm)
    else
        push!(γd_E, 0.0);
        push!(γd_N, 0.0);
        push!(γd_D, 1.0)
    end

    # Reflected wave: image source at depth (2*INTERFACE_DEPTH - event_depth)
    z_image = 2 * INTERFACE_DEPTH - event_depth  # 30 km
    rh_ref = sqrt(d_km^2 + z_image^2)
    push!(tp_ref_sec, rh_ref / VP_UPPER)
    push!(ts_ref_sec, rh_ref / VS_UPPER)

    # Reflected wave direction cosines (source → interface, downward leg)
    # Approximate: direction from source to midpoint of reflected path
    # The reflection point is at horizontal offset d_km/3 from source (by image method)
    # For the downward leg direction, use the same horizontal component as direct
    # but with D component = +(INTERFACE_DEPTH - event_depth) (positive down)
    # Actually use image source γ for simplicity — same horizontal, D = +z_image
    vr_ve = lons[i] * 111.0 * 1000.0
    vr_vn = lats[i] * 111.0 * 1000.0
    vr_vd = z_image * 1000.0  # positive down
    vr_norm = sqrt(vr_ve^2 + vr_vn^2 + vr_vd^2)
    if vr_norm > 0
        push!(γr_E, vr_ve / vr_norm)
        push!(γr_N, vr_vn / vr_norm)
        push!(γr_D, vr_vd / vr_norm)
    else
        push!(γr_E, 0.0);
        push!(γr_N, 0.0);
        push!(γr_D, 1.0)
    end

    # Phase picks: use DIRECT P and S times only
    p_time = origin_dt + Millisecond(round(Int, tp_dir_sec[i] * 1000))
    s_time = origin_dt + Millisecond(round(Int, ts_dir_sec[i] * 1000))
    push!(P_times, Dates.format(p_time, "yyyy-mm-ddTHH:MM:SS"))
    push!(S_times, Dates.format(s_time, "yyyy-mm-ddTHH:MM:SS"))
end

# ---------------------------------------------------------------------------
# 5. Green's functions: delta at direct + reflected arrivals
# ---------------------------------------------------------------------------

# MT pair indices (j,k) in NED: 1=N, 2=E, 3=D
# GF array channel order: [N, E, D] (index 1=N, 2=E, 3=D)
MT_PAIRS = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

function add_phase!(gf, nt, dt, idx, r_km, γ, scale, v)
    if r_km < 0.001 || idx < 1 || idx > nt
        return
    end
    amp = scale / r_km / v^3
    for (m, (j, k)) in enumerate(MT_PAIRS)
        for i in 1:3
            # P-wave from image: u_i = γ_i * γ_j * γ_k
            gf[idx, m, i] += amp * γ[i] * γ[j] * γ[k]
        end
    end
end

function add_reflected_phase!(gf, nt, dt, idx, r_km, γ, scale, v, R)
    add_phase!(gf, nt, dt, idx, r_km, γ, scale * R, v)
end

# Build GF for each station
gf_dict = Dict{Int, Array{Float64, 3}}()

for si in 1:n_station
    gf = zeros(Float64, npts, 6, 3)
    d_km = r_km[si]

    # Direct P
    tp_d = tp_dir_sec[si]
    tp_d_idx = max(1, min(npts, round(Int, tp_d / dt)))
    γd = [γd_N[si], γd_E[si], γd_D[si]]  # [N, E, D] order for pipeline convention
    add_phase!(gf, npts, dt, tp_d_idx, d_km, γd, AMPLITUDE_SCALE, VP_UPPER)

    # Direct S
    if d_km >= 0.001
        ts_d = ts_dir_sec[si]
        ts_d_idx = max(1, min(npts, round(Int, ts_d / dt)))
        s_scale = AMPLITUDE_SCALE / d_km / VS_UPPER^3
        for (m, (j, k)) in enumerate(MT_PAIRS)
            for i in 1:3
                δ_ij = i == j ? 1.0 : 0.0
                gf[ts_d_idx, m, i] += s_scale * (δ_ij - γd[i] * γd[j]) * γd[k]
            end
        end
    end

    # Reflected P
    tp_r = tp_ref_sec[si]
    tp_r_idx = max(1, min(npts, round(Int, tp_r / dt)))
    γr = [γr_N[si], γr_E[si], γr_D[si]]  # [N, E, D] order
    add_reflected_phase!(gf, npts, dt, tp_r_idx, d_km, γr, AMPLITUDE_SCALE, VP_UPPER, R_PP)

    # Reflected S
    if d_km >= 0.001
        ts_r = ts_ref_sec[si]
        ts_r_idx = max(1, min(npts, round(Int, ts_r / dt)))
        r_scale = AMPLITUDE_SCALE * R_SS / d_km / VS_UPPER^3
        for (m, (j, k)) in enumerate(MT_PAIRS)
            for i in 1:3
                δ_ij = i == j ? 1.0 : 0.0
                gf[ts_r_idx, m, i] += r_scale * (δ_ij - γr[i] * γr[j]) * γr[k]
            end
        end
    end

    gf_dict[si] = gf
end

# ---------------------------------------------------------------------------
# 6. Synthetic observed waveforms: obs = GF * MT + noise
# ---------------------------------------------------------------------------

CH_NAMES = ["E", "N", "Z"]
# For each output channel (E,N,Z), GF index 1=N, 2=E, 3=D
const _CH_TO_GF = [2, 1, 3]

waveforms = Dict{String, Vector{Float64}}()
noise_rng = Random.MersenneTwister(999)

for si in 1:n_station
    gf = gf_dict[si]
    for (oci, ch_name) in enumerate(CH_NAMES)
        gf_ch = _CH_TO_GF[oci]
        ch_id = sta_ids[si] * "." * ch_name
        syn = zeros(Float64, npts)
        for m in 1:6
            syn .+= gf[:, m, gf_ch] .* mt_true[m]
        end
        # Z channel: output positive up (seismic convention), flip from GF D-down
        if ch_name == "Z"
            syn .= -syn
        end
        rms = sqrt(sum(syn .^ 2) / npts)
        noise = randn(noise_rng, Float64, npts) .* (rms * 0.1)
        waveforms[ch_id] = syn .+ noise
    end
end

# ---------------------------------------------------------------------------
# 7. Write output files
# ---------------------------------------------------------------------------

# 7a. Station list
open(joinpath(outdir, "stations.txt"), "w") do f
    write(f, "# station_id  lat  lon\n")
    for i in 1:n_station
        write(f, "$(sta_ids[i])  $(lats[i])  $(lons[i])\n")
    end
end

# 7b. Waveform files
for ch_id in sort(collect(keys(waveforms)))
    data = waveforms[ch_id]
    fn = ch_id * ".dat"
    open(joinpath(outdir, fn), "w") do f
        for v in data
            write(f, "$v\n")
        end
    end
end

# 7c. Phase picks
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
println("  stations.txt  — $(n_station) stations")
println("  phases.txt    — phase picks")
for ch_id in sort(collect(keys(waveforms)))
    println("  $(ch_id).dat  — waveform ($npts samples)")
end
println("  SDR  strike=$(strike)  dip=$(dip)  rake=$(rake)")
mt_str = join(round.(mt_true, digits = 4), ", ")
println("  MT   [$mt_str]")
println("  event  depth=$(event_depth) km  at origin")
println(
    "  velocity model: upper(0–20 km) vp=$(VP_UPPER) vs=$(VS_UPPER), lower vp=$(VP_LOWER) vs=$(VS_LOWER)",
)
println("  reflection coeff: P=$(round(R_PP, digits=4)) S=$(round(R_SS, digits=4))")
