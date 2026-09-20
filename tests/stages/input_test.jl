# input_test.jl — Stage 1 (input) schema tests.
#
# Runs `scripts/input.jl` on a fresh synthetic event and asserts the
# `database.h5`/`status_0.h5` schema (see doc/schema.md): /paraspace holds
# values, /config params, /strategy indices.
#
# Usage:
#   julia --project=. tests/stages/input_test.jl

using Test
using HDF5
using IO

include("test_util.jl")

# ── XCorr P+S config (mirrors examples/synthetic/config.jl) ──────────────
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
                    @test read(ps["duration"]) == [0.1, 0.2, 0.3]
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

                    # Direct-wave GF must contract the six unique MT components
                    # exactly like the full symmetric tensor.
                    lat = read(f["/station/latitude"])[1]
                    lon = read(f["/station/longitude"])[1]
                    d_km = IO.haversine_distance(0.0, 0.0, lat, lon)
                    ve = lon * 111.0 * 1000.0 * cosd(lat / 2)
                    vn = lat * 111.0 * 1000.0
                    vd = -10.0 * 1000.0
                    γ = [vn, ve, vd] ./ sqrt(vn^2 + ve^2 + vd^2)
                    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
                    tp_idx = round(Int, sqrt(d_km^2 + 10.0^2) / 6.0 / 0.01)
                    ts_idx = round(Int, sqrt(d_km^2 + 10.0^2) / 4.0 / 0.01)

                    for (ch, i, sign) in (("N", 1, 1.0), ("E", 2, 1.0), ("Z", 3, -1.0))
                        m = read(gf["2"]["NET.ST1.$ch"])
                        p_expected = Float64[]
                        s_expected = Float64[]
                        for (j, k) in pairs
                            pair_weight = j == k ? 1.0 : 2.0
                            push!(p_expected, pair_weight * γ[i] * γ[j] * γ[k])
                            s_coeff = ((i == j ? 1.0 : 0.0) - γ[i] * γ[j]) * γ[k]
                            if j != k
                                s_coeff += ((i == k ? 1.0 : 0.0) - γ[i] * γ[k]) * γ[j]
                            end
                            push!(s_expected, s_coeff)
                        end
                        p_expected .*= sign * 1.0e6 / d_km / 6.0^3
                        s_expected .*= sign * 1.0e6 / d_km / 4.0^3
                        @test m[tp_idx, :] ≈ p_expected rtol = 1.0e-12
                        @test m[ts_idx, :] ≈ s_expected rtol = 1.0e-12
                    end
                end

                @testset "/config" begin
                    cf = f["/config"]
                    @test String.(read(cf["misfit_modules"])) ==
                          ["XcorrP", "XcorrS", "LagP", "LagS", "Psr", "PolarityP"]
                    for (name, phase, output) in (
                        ("XcorrP", "P", "cc_max"),
                        ("XcorrS", "S", "cc_max"),
                        ("LagP", "P", "best_lag"),
                        ("LagS", "S", "best_lag"),
                    )
                        x = cf[name]
                        @test read(x["trim"]) == [-2.0, 8.0]
                        @test read(x["max_lag_periods"]) ≈ 3.0
                        @test read(x["filter_order"]) == 4
                        @test read(x["band_low"]) == [1]
                        @test read(x["band_high"]) == [2]
                        @test String(read(x["operator"])) == "Xcorr"
                        @test String(read(x["output"])) == output
                        @test read(x["is_composed"]) == 0
                        @test String(read(x["phase"])) == phase
                        @test String(read(x["channel"])) == ""
                    end
                    @test String(read(cf["Psr/operator"])) == "Expression"
                    @test String(read(cf["Psr/output"])) == "value"
                    @test String.(read(cf["Psr/bases"])) == ["XcorrP", "XcorrS"]
                    @test read(cf["Psr/is_composed"]) == 1
                    @test String(read(cf["PolarityP/operator"])) == "Expression"
                    @test String(read(cf["PolarityP/output"])) == "value"
                    @test String.(read(cf["PolarityP/bases"])) == ["XcorrP"]
                    @test read(cf["PolarityP/is_composed"]) == 1
                end

                for name in ("XcorrP", "XcorrS")
                    @testset "/$name" begin
                        x = f["/$name"]
                        ch_entries = String.(read(x["channel_id"]))
                        sta_idx = read(x["station_idx"])
                        n_entries = length(ch_entries)

                        @test n_entries >= 1
                        @test length(sta_idx) == n_entries
                        # station_idx points into /station (1-based), all valid
                        n_phys = length(read(f["/station"]["id"]))
                        @test all(1 .<= sta_idx .<= Int32(n_phys))
                        @test all([
                            startswith(ch_entries[i], "NET.ST$(Int(sta_idx[i])).") for
                            i in 1:n_entries
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
                            for duration_idx in ("1", "2", "3")
                                sl = read(x["synamp_lag"][d]["1"][duration_idx])
                                @test size(sl) == (n_entries, 6, 6, 301)
                                gf_arr = read(x["gf"][d]["1"][duration_idx]["gf"])
                                @test size(gf_arr) == (n_entries, 6, 501)
                            end
                        end
                        for duration_idx in ("1", "2", "3")
                            dog = read(x["dot_obs_gf_lag"]["1"][duration_idx])
                            @test size(dog) == (n_entries, 6, 301)
                        end
                    end
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
                @test read(st["duration_indices"]) == [1, 2, 3]
                @test read(st["iteration"]) == 0
            end
        end
    end
end
