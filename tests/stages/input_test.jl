# input_test.jl — Stage 1 (input) schema tests.
#
# Runs `scripts/input.jl` as a subprocess against a freshly generated synthetic
# event, then asserts the `database.h5` and `status_0.h5` schema against the
# contract in `doc/schema.md` (three-layer separation: /paraspace holds values,
# /config holds params without indices, /strategy holds indices).
#
# Usage:
#   julia --project=. tests/stages/input_test.jl

using Test
using HDF5

include("test_util.jl")

# ---------------------------------------------------------------------------
# Minimal XCorrS-only config, mirroring examples/synthetic/config.jl.
# ---------------------------------------------------------------------------
function write_test_config(dir::AbstractString)
    cfg = """
    # test config.jl — XCorrS-only, mirrors examples/synthetic.
    using Random
    using Misfit

    Config.use_misfit!(:XcorrS, operator = Misfit.Xcorr, phase = "S", output = Misfit.Xcorr.CC_MAX)

    Config.XcorrS.trim() = [-2.0, 8.0]
    Config.XcorrS.max_lag_periods() = 3.0
    Config.XcorrS.filter_order() = 4
    Config.XcorrS.band_low() = Int32[1]
    Config.XcorrS.band_high() = Int32[2]

    Config.freq_bands() = [(0.5, 2.0)]
    Config.depths() = [5.0, 10.0, 15.0]

    Config.phase_fields() = Dict("P" => :P_time, "S" => :S_time)
    Config.polarity_fields() = Dict("P" => :P_polarity)

    const _DIR = "$dir"
    const _CH_NAMES = ["E", "N", "Z"]

    Config.load_event() = IO.EventInfo(
        0.0, 0.0, 10.0, 1.0, "2024-01-01T00:00:00")

    Config.load_stations() = begin
        stas = IO.StationInfo[]
        open(joinpath(_DIR, "stations.txt")) do f
            for line in eachline(f)
                startswith(line, '#') && continue
                parts = split(line)
                length(parts) < 3 && continue
                sid = parts[1]
                lat = parse(Float64, parts[2])
                lon = parse(Float64, parts[3])
                net_sta = split(sid, ".")
                net = net_sta[1]
                sta = join(net_sta[2:end], ".")
                for ch in _CH_NAMES
                    push!(stas, IO.StationInfo(
                        sid, net, sta, ch, lat, lon, 0.0, 0.01,
                        "2024-01-01T00:00:00"))
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
        parts = split(pid, ".")
        ch_id = join(parts[1:3], ".")
        fn = joinpath(_DIR, "\$ch_id.dat")
        return [parse(Float64, line) for line in eachline(fn)]
    end

    # Two-layer half-space GF physics — same as tests/synthetic_data.jl and
    # examples/synthetic/config.jl.
    const _INTERFACE_DEPTH = 20.0
    const _VP_UPPER = 6.0
    const _VS_UPPER = 4.0
    const _VP_LOWER = 8.0
    const _VS_LOWER = 6.0
    const _R_PP = (_VP_LOWER - _VP_UPPER) / (_VP_LOWER + _VP_UPPER)
    const _R_SS = (_VS_LOWER - _VS_UPPER) / (_VS_LOWER + _VS_UPPER)
    const _AMP_SCALE = 1.0e6
    const _MT_PAIRS = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    _add_p(gf, idx, r_km, γ, scale, v, nt) = begin
        (r_km < 0.001 || idx < 1 || idx > nt) && return
        amp = scale / r_km / v^3
        for (m, (j, k)) in enumerate(_MT_PAIRS)
            for i in 1:3
                gf[idx, m, i] += amp * γ[i] * γ[j] * γ[k]
            end
        end
    end

    _add_s(gf, idx, r_km, γ, scale, v, nt) = begin
        (r_km < 0.001 || idx < 1 || idx > nt) && return
        amp = scale / r_km / v^3
        for (m, (j, k)) in enumerate(_MT_PAIRS)
            for i in 1:3
                δ_ij = i == j ? 1.0 : 0.0
                gf[idx, m, i] += amp * (δ_ij - γ[i] * γ[j]) * γ[k]
            end
        end
    end

    Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
        nt = 2000
        dt = 0.01
        d_km = IO.haversine_distance(src_lat, src_lon, sta_lat, sta_lon)
        d_km = max(d_km, 0.001)

        rh_dir = sqrt(d_km^2 + src_depth^2)
        tp_dir = rh_dir / _VP_UPPER
        ts_dir = rh_dir / _VS_UPPER

        z_image = 2 * _INTERFACE_DEPTH - src_depth
        rh_ref = sqrt(d_km^2 + z_image^2)
        tp_ref = rh_ref / _VP_UPPER
        ts_ref = rh_ref / _VS_UPPER

        dlat_deg = sta_lat - src_lat
        dlon_deg = sta_lon - src_lon
        avg_lat = (src_lat + sta_lat) / 2 * pi / 180
        ve = dlon_deg * 111.0 * 1000.0 * cos(avg_lat)
        vn = dlat_deg * 111.0 * 1000.0
        vd = -src_depth * 1000.0
        vnorm = sqrt(ve^2 + vn^2 + vd^2)
        γd = vnorm < 1.0 ? [0.0, 0.0, -1.0] : [vn / vnorm, ve / vnorm, vd / vnorm]

        vr_ve = dlon_deg * 111.0 * 1000.0 * cos(avg_lat)
        vr_vn = dlat_deg * 111.0 * 1000.0
        vr_vd = z_image * 1000.0
        vr_norm = sqrt(vr_ve^2 + vr_vn^2 + vr_vd^2)
        γr = vr_norm < 1.0 ? [0.0, 0.0, 1.0] : [vr_vn / vr_norm, vr_ve / vr_norm, vr_vd / vr_norm]

        gf = zeros(Float64, nt, 6, 3)
        _add_p(gf, max(1, min(nt, round(Int, tp_dir / dt))), d_km, γd, _AMP_SCALE, _VP_UPPER, nt)
        _add_s(gf, max(1, min(nt, round(Int, ts_dir / dt))), d_km, γd, _AMP_SCALE, _VS_UPPER, nt)
        _add_p(gf, max(1, min(nt, round(Int, tp_ref / dt))), d_km, γr, _AMP_SCALE * _R_PP, _VP_UPPER, nt)
        _add_s(gf, max(1, min(nt, round(Int, ts_ref / dt))), d_km, γr, _AMP_SCALE * _R_SS, _VS_UPPER, nt)

        gf[:, :, 3] .= -gf[:, :, 3]
        return (gf, dt, tp_dir, ts_dir)
    end
    """
    write(joinpath(dir, "config.jl"), cfg)
    return joinpath(dir, "config.jl")
end

@testset "input stage" begin
    mktempdir() do dir
        nsta = 3
        make_synthetic(dir; nsta = nsta)
        config = write_test_config(dir)

        r = run_stage_script(joinpath("scripts", "input.jl"), [config])
        @testset "stage run" begin
            @test r.ok
            @test isfile(joinpath(dir, "database.h5"))
            @test isfile(joinpath(dir, "status_0.h5"))
        end

        db = joinpath(dir, "database.h5")
        status0 = joinpath(dir, "status_0.h5")

        @testset "database.h5 schema" begin
            h5open(db, "r") do f
                @testset "/event" begin
                    ev = f["/event"]
                    @test haskey(ev, "longitude") && read(ev["longitude"]) ≈ 0.0
                    @test haskey(ev, "latitude") && read(ev["latitude"]) ≈ 0.0
                    @test haskey(ev, "depth") && read(ev["depth"]) ≈ 10.0
                    @test haskey(ev, "magnitude") && read(ev["magnitude"]) ≈ 1.0
                    @test haskey(ev, "origintime")
                    @test String(read(ev["origintime"])) == "2024-01-01T00:00:00"
                end

                @testset "/paraspace" begin
                    ps = f["/paraspace"]
                    @test read(ps["strike"]) == collect(0.0:5.0:355.0)
                    @test read(ps["dip"]) == collect(0.0:5.0:90.0)
                    @test read(ps["rake"]) == collect(-90.0:5.0:90.0)
                    @test read(ps["depth"]) == [5.0, 10.0, 15.0]
                    @test read(ps["frequency"]) == [0.5, 2.0]
                end

                @testset "/station" begin
                    st = f["/station"]
                    expected_keys = [
                        "id",
                        "network",
                        "station",
                        "latitude",
                        "longitude",
                        "elevation",
                        "dt",
                        "begin_time",
                        "distance",
                        "azimuth",
                        "P_time",
                        "S_time",
                        "P_polarity",
                    ]
                    @test Set(string.(keys(st))) == Set(expected_keys)
                    ids = String.(read(st["id"]))
                    @test length(ids) == nsta
                    @test ids == ["NET.ST$i" for i in 1:nsta]  # dedup keeps original order
                    @test all(read(st["network"]) .== "NET")
                    @test all(read(st["dt"]) .≈ 0.01)
                end

                @testset "/channel" begin
                    ch = f["/channel"]
                    @test length(keys(ch)) == 3 * nsta
                    for sta in 1:nsta
                        for c in ("E", "N", "Z")
                            key = "NET.ST$sta.$c"
                            @test haskey(ch, key)
                            @test length(read(ch[key])) == 2000
                        end
                    end
                end

                @testset "/gf" begin
                    gf = f["/gf"]
                    @test Set(string.(keys(gf))) == Set(["1", "2", "3"])
                    for d in ("1", "2", "3")
                        gd = gf[d]
                        @test length(keys(gd)) == 3 * nsta
                        for sta in 1:nsta
                            for c in ("E", "N", "Z")
                                m = read(gd["NET.ST$sta.$c"])
                                @test size(m) == (2000, 6)
                            end
                        end
                    end
                end

                @testset "/config" begin
                    cf = f["/config"]
                    @test String.(read(cf["misfit_modules"])) == ["XcorrS"]
                    x = cf["XcorrS"]
                    @test read(x["trim"]) == [-2.0, 8.0]
                    @test read(x["max_lag_periods"]) ≈ 3.0
                    @test read(x["filter_order"]) == 4
                    @test read(x["band_low"]) == [1]
                    @test read(x["band_high"]) == [2]
                    @test String(read(x["operator"])) == "Xcorr"
                    @test String(read(x["output"])) == "cc_max"
                    @test read(x["is_composed"]) == 0
                    @test String(read(x["phase"])) == "S"
                    @test String(read(x["channel"])) == ""
                end

                @testset "/XcorrS" begin
                    x = f["/XcorrS"]
                    ch_entries = String.(read(x["channel_id"]))
                    sta_idx = read(x["station_idx"])
                    n_entries = length(ch_entries)

                    @test n_entries >= 1
                    @test length(sta_idx) == n_entries
                    # station_idx points into /station (1-based), all valid
                    n_phys = length(read(f["/station"]["id"]))
                    @test all(1 .<= sta_idx .<= Int32(n_phys))
                    @test all([
                        startswith(ch_entries[i], "NET.ST$(Int(sta_idx[i])).") for i in 1:n_entries
                    ],)

                    # obs: [N_entries, nt_win=501], obs_norm2: [N_entries]
                    obs = read(x["obs"]["1"]["obs"])
                    obs_norm2 = read(x["obs"]["1"]["obs_norm2"])
                    @test size(obs) == (n_entries, 501)
                    @test length(obs_norm2) == n_entries
                    @test all(obs_norm2 .> 0.0)
                    @test all(obs_norm2 .≈ vec(sum(abs2, obs; dims = 2)))

                    # per-lag shapes: L = 2*150 + 1 = 301
                    for d in ("1", "2", "3")
                        sl = read(x["synamp_lag"][d]["1"])
                        @test size(sl) == (n_entries, 6, 6, 301)
                        gf_arr = read(x["gf"][d]["1"]["gf"])
                        @test size(gf_arr) == (n_entries, 6, 501)
                    end
                    dog = read(x["dot_obs_gf_lag"]["1"])
                    @test size(dog) == (n_entries, 6, 301)
                end
            end
        end

        @testset "status_0.h5 schema" begin
            h5open(status0, "r") do f
                st = f["/strategy"]
                @test read(st["strike0"]) ≈ 0.0
                @test read(st["dstrike"]) ≈ 5.0
                @test read(st["nstrike"]) == 72
                @test read(st["dip0"]) ≈ 0.0
                @test read(st["ddip"]) ≈ 5.0
                @test read(st["ndip"]) == 19
                @test read(st["rake0"]) ≈ -90.0
                @test read(st["drake"]) ≈ 5.0
                @test read(st["nrake"]) == 37
                @test read(st["depth_indices"]) == [1, 2, 3]
                @test read(st["freq_indices"]) == [1]
                @test read(st["iteration"]) == 0
            end
        end
    end
end
