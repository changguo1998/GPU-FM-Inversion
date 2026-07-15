# config.jl — Synthetic test event (plain-text data, Gaussian STF)
# Loaded by input.jl via include() — Config module already loaded.
# Reads stations.txt, {sta}.{ch}.dat, phases.txt from @__DIR__.
#
# Physics matches tests/synthetic_data.jl: two-layer half-space,
# far-field P/S waves with Gaussian source time function (σ=0.2s).

using Random

Config.use_misfit!(:XcorrP, from = :Xcorr, phase_type = "P")
Config.use_misfit!(:XcorrS, from = :Xcorr, phase_type = "S")
Config.use_misfit!(:PolarityP, from = :Polarity, phase_type = "P")

Config.XcorrP.trim() = [-2.0, 5.0]
Config.XcorrP.maxlag_factor() = 0.5
Config.XcorrP.filter_order() = 4
Config.XcorrP.select_threshold() = 0.5
Config.XcorrP.deselect_threshold() = 0.3
Config.XcorrP.band_low() = Int32[1]
Config.XcorrP.band_high() = Int32[2]

Config.XcorrS.trim() = [-2.0, 8.0]
Config.XcorrS.maxlag_factor() = 0.5
Config.XcorrS.filter_order() = 4
Config.XcorrS.select_threshold() = 0.5
Config.XcorrS.deselect_threshold() = 0.3
Config.XcorrS.band_low() = Int32[1]
Config.XcorrS.band_high() = Int32[2]

Config.PolarityP.trim() = [0.0, 2.0]

Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 10.0, 15.0]

Config.phase_fields() = Dict("P" => :P_time, "S" => :S_time)
Config.polarity_fields() = Dict("P" => :P_polarity)

# ---------------------------------------------------------------------------
# Data directory = config file location
# ---------------------------------------------------------------------------
const _DIR = @__DIR__
const _CH_NAMES = ["E", "N", "Z"]

# Waveform files: E positive east, N positive north, Z positive up.
# GF array channels: [N, E, Z] (Z positive up) — flipped from internal NED D-down.

# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

Config.load_event() = IO.EventInfo(
    0.0,   # longitude
    0.0,   # latitude
    10.0,  # depth (km)
    1.0,   # magnitude
    "2024-01-01T00:00:00",  # origin time
)

Config.load_stations() = begin
    stas = IO.StationInfo[]
    open(joinpath(_DIR, "stations.txt")) do f
        for line in eachline(f)
            startswith(line, '#') && continue
            parts = split(line)
            length(parts) < 3 && continue
            sid = parts[1]                   # "NET.ST1"
            lat = parse(Float64, parts[2])
            lon = parse(Float64, parts[3])
            net_sta = split(sid, ".")
            net = net_sta[1]                 # "NET"
            sta = join(net_sta[2:end], ".")  # "ST1"
            for ch in _CH_NAMES
                push!(
                    stas,
                    IO.StationInfo(
                        sid,
                        net,
                        sta,
                        ch,
                        lat,
                        lon,
                        0.0,      # elevation
                        0.01,     # dt (100 Hz)
                        "2024-01-01T00:00:00",  # begin_time
                    ),
                )
            end
        end
    end
    return stas
end

Config.load_phase_picks() = begin
    picks = IO.PhasePick[]
    open(joinpath(_DIR, "phases.txt")) do f
        for line in eachline(f)
            startswith(line, '#') && continue
            parts = split(line)
            length(parts) < 3 && continue
            push!(picks, IO.PhasePick(parts[1], parts[2], parts[3], Int8(0)))
        end
    end
    return picks
end

Config.load_waveform(pid::String) = begin
    # pid = "NET.ST1.E.P" → strip phase type → "NET.ST1.E" → "NET.ST1.E.dat"
    parts = split(pid, ".")
    ch_id = join(parts[1:3], ".")
    fn = joinpath(_DIR, "$ch_id.dat")
    return [parse(Float64, line) for line in eachline(fn)]
end

# ---------------------------------------------------------------------------
# GF physics helpers — internal computation in NED (D positive down)
# Result flipped to Z-up at end of load_gf to match observed waveform convention
# ---------------------------------------------------------------------------

const _INTERFACE_DEPTH = 20.0   # km
const _VP_UPPER = 6.0           # km/s
const _VS_UPPER = 4.0           # km/s
const _VP_LOWER = 8.0           # km/s
const _VS_LOWER = 6.0           # km/s
const _R_PP = (_VP_LOWER - _VP_UPPER) / (_VP_LOWER + _VP_UPPER)   # ≈ 0.143
const _R_SS = (_VS_LOWER - _VS_UPPER) / (_VS_LOWER + _VS_UPPER)   # ≈ 0.2
const _AMP_SCALE = 1.0e6
const _MT_PAIRS = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

_add_p(gf, idx, r_km, γ, scale, v, nt) = begin
    if r_km < 0.001 || idx < 1 || idx > nt
        return
    end
    amp = scale / r_km / v^3
    for (m, (j, k)) in enumerate(_MT_PAIRS)
        for i in 1:3
            gf[idx, m, i] += amp * γ[i] * γ[j] * γ[k]
        end
    end
end

_add_s(gf, idx, r_km, γ, scale, v, nt) = begin
    if r_km < 0.001 || idx < 1 || idx > nt
        return
    end
    amp = scale / r_km / v^3
    for (m, (j, k)) in enumerate(_MT_PAIRS)
        for i in 1:3
            δ_ij = i == j ? 1.0 : 0.0
            gf[idx, m, i] += amp * (δ_ij - γ[i] * γ[j]) * γ[k]
        end
    end
end

# ---------------------------------------------------------------------------
# Green's function — two-layer full-space, P/S delta impulses
# ---------------------------------------------------------------------------

Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
    nt = 2000
    dt = 0.01

    # Distance
    d_km = IO.haversine_distance(src_lat, src_lon, sta_lat, sta_lon)
    d_km = max(d_km, 0.001)

    # Travel times
    rh_dir = sqrt(d_km^2 + src_depth^2)
    tp_dir = rh_dir / _VP_UPPER
    ts_dir = rh_dir / _VS_UPPER

    z_image = 2 * _INTERFACE_DEPTH - src_depth
    rh_ref = sqrt(d_km^2 + z_image^2)
    tp_ref = rh_ref / _VP_UPPER
    ts_ref = rh_ref / _VS_UPPER

    # Direction cosines (source → receiver, NED components)
    dlat_deg = sta_lat - src_lat
    dlon_deg = sta_lon - src_lon
    avg_lat = (src_lat + sta_lat) / 2 * pi / 180
    ve = dlon_deg * 111.0 * 1000.0 * cos(avg_lat)  # east (m)
    vn = dlat_deg * 111.0 * 1000.0                  # north (m)
    vd = -src_depth * 1000.0                        # upward = negative D
    vnorm = sqrt(ve^2 + vn^2 + vd^2)
    if vnorm < 1.0
        γd = [0.0, 0.0, -1.0]
    else
        γd = [vn / vnorm, ve / vnorm, vd / vnorm]  # [N, E, D] order
    end

    # Image source → receiver vector
    vr_ve = dlon_deg * 111.0 * 1000.0 * cos(avg_lat)
    vr_vn = dlat_deg * 111.0 * 1000.0
    vr_vd = z_image * 1000.0
    vr_norm = sqrt(vr_ve^2 + vr_vn^2 + vr_vd^2)
    if vr_norm < 1.0
        γr = [0.0, 0.0, 1.0]
    else
        γr = [vr_vn / vr_norm, vr_ve / vr_norm, vr_vd / vr_norm]
    end

    # Build GF (delta impulses at arrival times)
    gf = zeros(Float64, nt, 6, 3)

    tp_d_idx = max(1, min(nt, round(Int, tp_dir / dt)))
    _add_p(gf, tp_d_idx, d_km, γd, _AMP_SCALE, _VP_UPPER, nt)

    ts_d_idx = max(1, min(nt, round(Int, ts_dir / dt)))
    _add_s(gf, ts_d_idx, d_km, γd, _AMP_SCALE, _VS_UPPER, nt)

    tp_r_idx = max(1, min(nt, round(Int, tp_ref / dt)))
    _add_p(gf, tp_r_idx, d_km, γr, _AMP_SCALE * _R_PP, _VP_UPPER, nt)

    ts_r_idx = max(1, min(nt, round(Int, ts_ref / dt)))
    _add_s(gf, ts_r_idx, d_km, γr, _AMP_SCALE * _R_SS, _VS_UPPER, nt)

    # GF computed in NED (D positive down); observed Z = positive up → flip D to Z
    gf[:, :, 3] .= -gf[:, :, 3]

    return (gf, dt, tp_dir, ts_dir)
end
