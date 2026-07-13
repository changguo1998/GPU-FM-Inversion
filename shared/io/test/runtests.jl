using Test, HDF5, IO

tmpfile(fn) = joinpath(@__DIR__, fn)

function make_synthetic_event()
    HDF5.h5open(tmpfile("test_event.h5"), "w") do f
        gr = HDF5.create_group(f, "event")
        write(gr, "longitude", 118.5)
        write(gr, "latitude", 32.1)
        write(gr, "depth", 12.3)
        write(gr, "magnitude", 4.7)
        write(gr, "origintime", "2024-03-15T08:22:00")
    end
end

function make_synthetic_phase_picks()
    HDF5.h5open(tmpfile("test_phase_picks.h5"), "w") do f
        gr = HDF5.create_group(f, "phase_picks")
        write(gr, "station_ids", ["NET.STA1", "NET.STA2", "NET.STA3"])
        write(gr, "P_time", ["2024-03-15T08:22:15", "2024-03-15T08:22:18", ""])
        write(gr, "S_time", ["2024-03-15T08:22:40", "", "2024-03-15T08:22:55"])
        write(gr, "P_polarity", Int8[1, -1, 0])
    end
end

function make_synthetic_stations()
    HDF5.h5open(tmpfile("test_stations.h5"), "w") do f
        gr = HDF5.create_group(f, "stations")
        write(gr, "id", ["NET.STA1.BHE.P", "NET.STA1.BHN.S", "NET.STA2.BHE.P"])
        write(gr, "network", ["NET", "NET", "NET"])
        write(gr, "station", ["STA1", "STA1", "STA2"])
        write(gr, "channel", ["E", "N", "E"])
        write(gr, "latitude", [32.0, 32.0, 32.5])
        write(gr, "longitude", [118.0, 118.0, 117.5])
        write(gr, "elevation", [150.0, 150.0, 200.0])
        write(gr, "dt", [0.01, 0.01, 0.02])
        write(
            gr,
            "begin_time",
            ["2024-03-15T08:21:00", "2024-03-15T08:21:00", "2024-03-15T08:21:00"],
        )
    end
end

function make_synthetic_waveforms()
    HDF5.h5open(tmpfile("test_waveforms.h5"), "w") do f
        gr = HDF5.create_group(f, "waveforms")
        write(gr, "NET.STA1.BHE.P", Float64[1.0, 2.0, 3.0, 4.0, 5.0])
        write(gr, "NET.STA1.BHN.S", Float64[0.5, 1.5, 2.5])
    end
end

function make_synthetic_status()
    fn = tmpfile("test_status.h5")

    # ---- Strategy ----
    strategy = IO.Strategy(
        Int32[1, 2, 3],   # depth_indices
        Int32[1, 3],       # freq_low_idx
        Int32[2, 4],       # freq_high_idx
        Int32(3),           # iteration
    )

    # ---- Trials ----
    trials = IO.TrialSet(
        collect(100.0:10.0:200.0),  # strike
        collect(40.0:5.0:60.0),     # dip
        collect(-120.0:20.0:-60.0), # rake
        fill(12.3, 11),             # depth
        Int32.(1:11),               # depth_idx
        fill(Int32(1), 11),         # freq_idx
    )

    # ---- Misfits ----
    xcorr_misfit = rand(6, 11)
    polarity_misfit = rand(3, 11) .* 0.5
    psr_misfit = rand(3, 11) .* 0.3

    # Write everything
    HDF5.h5open(fn, "w") do f
        HDF5.create_group(f, "trials")
        HDF5.create_group(f, "strategy")
        HDF5.create_group(f, "misfits")
    end
    IO.write_strategy(fn, strategy)
    IO.write_trials(fn, trials)
    IO.write_misfits(fn, :xcorr, xcorr_misfit)
    IO.write_misfits(fn, :polarity, polarity_misfit)
    IO.write_misfits(fn, :psr, psr_misfit)

    return (; strategy, trials, xcorr_misfit, polarity_misfit, psr_misfit)
end

function make_synthetic_database()
    fn = tmpfile("test_database.h5")

    n_sta = 1
    ch_id = "NET.STA1.BHE"

    event = Dict(
        "longitude" => 118.5,
        "latitude" => 32.1,
        "depth" => 12.3,
        "magnitude" => 4.7,
        "origintime" => "2024-03-15T08:22:00",
    )

    station = Dict(
        "id" => ["NET.STA1"],
        "network" => ["NET"],
        "station" => ["STA1"],
        "channel" => ["BHE"],
        "latitude" => [32.0],
        "longitude" => [118.0],
        "elevation" => [150.0],
        "dt" => [0.01],
        "begin_time" => ["2024-03-15T08:21:00"],
        "distance" => [50.0],
        "azimuth" => [90.0],
        "P_time" => ["2024-03-15T08:22:15"],
        "S_time" => ["2024-03-15T08:22:40"],
        "P_polarity" => Int8[1],
    )

    channel_data = Dict(ch_id => rand(100))

    gf_data = Dict(
        5.0 => Dict(ch_id => rand(100, 6)),
        10.0 => Dict(ch_id => rand(100, 6)),
        15.0 => Dict(ch_id => rand(100, 6)),
    )


    config = Dict{String, Any}(
        "misfit_modules" => ["XcorrP", "XcorrS", "Polarity"],
        "n_bands" => Int32(2),
        "XcorrP" => Dict(
            "maxlag_factor" => 0.5,
            "filter_order" => Int32(4),
            "trim" => [-2.0, 60.0],
            "select_threshold" => 0.7,
            "deselect_threshold" => 0.5,
        ),
        "XcorrS" => Dict(
            "maxlag_factor" => 0.5,
            "filter_order" => Int32(4),
            "trim" => [-2.0, 80.0],
            "select_threshold" => 0.7,
            "deselect_threshold" => 0.5,
        ),
        "Polarity" => Dict("trim" => [0.0, 2.0]),
    )

    paraspace = Dict{String, Any}(
        "strike" => collect(0.0:5.0:355.0),
        "dip" => collect(0.0:5.0:90.0),
        "rake" => collect(-90.0:5.0:90.0),
        "depth" => [5.0, 10.0, 15.0],
        "frequency" => [0.05, 0.1, 0.5, 1.0],
    )

    # Build module_data from individual module data
    module_data = Dict{String, IO.ModuleData}()
    module_data["XcorrP"] = IO.ModuleData(
        obs = Dict("1" => rand(1, 80)),
        obs_norm2 = Dict("1" => [1.0]),
        gf = Dict(
            5.0 => Dict("1" => rand(1, 6, 80)),
            10.0 => Dict("1" => rand(1, 6, 80)),
            15.0 => Dict("1" => rand(1, 6, 80)),
        ),
        synamp = Dict(
            5.0 => Dict("1" => rand(6, 6, 1)),
            10.0 => Dict("1" => rand(6, 6, 1)),
            15.0 => Dict("1" => rand(6, 6, 1)),
        ),
    )
    module_data["XcorrS"] = IO.ModuleData(
        obs = Dict("1" => rand(1, 60)),
        obs_norm2 = Dict("1" => [1.0]),
        gf = Dict(
            5.0 => Dict("1" => rand(1, 6, 60)),
            10.0 => Dict("1" => rand(1, 6, 60)),
            15.0 => Dict("1" => rand(1, 6, 60)),
        ),
        synamp = Dict(
            5.0 => Dict("1" => rand(6, 6, 1)),
            10.0 => Dict("1" => rand(6, 6, 1)),
            15.0 => Dict("1" => rand(6, 6, 1)),
        ),
    )
    module_data["Polarity"] = IO.ModuleData(
        obs = Dict("1" => reshape(Float64[1.0, -1.0], 2, 1)),
        gf = Dict(
            5.0 => Dict("1" => rand(2, 6, 50)),
            10.0 => Dict("1" => rand(2, 6, 50)),
            15.0 => Dict("1" => rand(2, 6, 50)),
        ),
    )
    IO.write_database(
        fn,
        config,
        event,
        station,
        channel_data,
        gf_data,
        module_data;
        paraspace = paraspace,
    )
    return (; config, paraspace, ch_id)
end

function make_synthetic_output()
    fn = tmpfile("test_output.h5")

    solution = Dict(
        "strike" => 130.0,
        "dip" => 50.0,
        "rake" => -80.0,
        "depth" => 12.3,
        "moment_tensor" => [1.0, 2.0, 3.0, 0.5, -0.3, 1.2],
        "misfit" => 0.023,
    )

    uncertainty = Dict(
        "strike_std" => 5.0,
        "dip_std" => 3.0,
        "rake_std" => 7.0,
        "depth_range" => [10.0, 15.0],
        "freq_test_misfit_curve" => rand(2, 3),
    )

    per_phase = Dict(
        "phase_id" => ["NET.STA1.Z.P", "NET.STA2.Z.P", "NET.STA3.Z.S"],
        "channel_id" => ["NET.STA1.Z", "NET.STA2.Z", "NET.STA3.Z"],
        "station_id" => ["NET.STA1", "NET.STA2", "NET.STA3"],
        "phase_type" => ["P", "P", "S"],
        "misfit_per_module" => rand(3, 3),
        "selected" => Int32[1, 1, 0],
        "cross_correlation" => [0.85, 0.72, 0.0],
    )

    per_station_summary = Dict(
        "station_id" => ["NET.STA1", "NET.STA2", "NET.STA3"],
        "n_channels" => Int32[1, 1, 1],
        "n_phases" => Int32[1, 1, 1],
        "mean_cross_correlation" => [0.85, 0.72, 0.0],
        "polarity_match" => Int32[0, 0, 0],
        "misfit_total" => [0.1, 0.2, 0.3],
    )

    summary = Dict(
        "total_iterations" => Int32(3),
        "total_trials" => Int32(1500),
        "convergence_reason" => "user",
    )

    IO.write_output(fn, solution, uncertainty, per_phase, per_station_summary, summary)
    return (; solution, uncertainty, per_phase, per_station_summary, summary)
end

# ─────────────────────────────
# Tests
# ─────────────────────────────

@testset "IO" begin
    @testset "read_event" begin
        make_synthetic_event()
        evt = IO.read_event(tmpfile("test_event.h5"))
        @test evt.longitude ≈ 118.5
        @test evt.latitude ≈ 32.1
        @test evt.depth ≈ 12.3
        @test evt.magnitude ≈ 4.7
        @test evt.origintime == "2024-03-15T08:22:00"
    end

    @testset "read_phase_picks" begin
        make_synthetic_phase_picks()
        picks = IO.read_phase_picks(tmpfile("test_phase_picks.h5"))
        @test length(picks) == 3
        @test picks[1].station_id == "NET.STA1"
        @test picks[1].P_time == "2024-03-15T08:22:15"
        @test picks[1].S_time == "2024-03-15T08:22:40"
        @test picks[1].P_polarity == 1
        @test picks[2].P_polarity == -1
        @test picks[3].P_polarity == 0
        @test picks[2].S_time == ""
        @test picks[3].P_time == ""
    end

    @testset "read_stations" begin
        make_synthetic_stations()
        stas = IO.read_stations(tmpfile("test_stations.h5"))
        @test length(stas) == 3
        @test stas[1].id == "NET.STA1.BHE.P"
        @test stas[1].latitude ≈ 32.0
        @test stas[1].dt ≈ 0.01
        @test stas[3].elevation ≈ 200.0
    end

    @testset "read_waveform" begin
        make_synthetic_waveforms()
        wf = IO.read_waveform(tmpfile("test_waveforms.h5"), "NET.STA1.BHE.P")
        @test wf ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
        wf2 = IO.read_waveform(tmpfile("test_waveforms.h5"), "NET.STA1.BHN.S")
        @test wf2 ≈ [0.5, 1.5, 2.5]
    end

    @testset "strategy round-trip" begin
        expected = make_synthetic_status()
        fn = tmpfile("test_status.h5")

        # Read strategy
        strat = IO.read_strategy(fn)
        @test strat.depth_indices == Int32[1, 2, 3]
        @test strat.freq_low_idx == Int32[1, 3]
        @test strat.freq_high_idx == Int32[2, 4]
        @test strat.iteration == 3
    end

    @testset "trials round-trip" begin
        fn = tmpfile("test_status.h5")
        trials = IO.read_trials(fn)
        @test length(trials.strike) == 11
        @test trials.strike[1] ≈ 100.0
        @test trials.strike[end] ≈ 200.0
        @test trials.depth_idx isa Vector{Int32}
        @test all(trials.freq_idx .== 1)
    end

    @testset "misfits round-trip" begin
        fn = tmpfile("test_status.h5")
        mis = IO.read_misfits(fn)
        @test haskey(mis, :xcorr)
        @test haskey(mis, :polarity)
        @test haskey(mis, :psr)
        @test size(mis[:xcorr]) == (6, 11)
        @test size(mis[:polarity]) == (3, 11)
        @test size(mis[:psr]) == (3, 11)

        # Verify data values
        expected = make_synthetic_status()
        @test mis[:xcorr] ≈ expected.xcorr_misfit
        @test mis[:polarity] ≈ expected.polarity_misfit
        @test mis[:psr] ≈ expected.psr_misfit
    end

    @testset "database round-trip" begin
        ex = make_synthetic_database()
        fn = tmpfile("test_database.h5")

        # Read greens
        g = IO.read_greens(fn, "NET.STA1.BHE.P", Int32(1))
        @test size(g) == (100, 6)

        # Read config — no indices, no float param values
        cfg = IO.read_config(fn)
        @test cfg["misfit_modules"] == ["XcorrP", "XcorrS", "Polarity"]
        @test haskey(cfg, "XcorrP")
        @test cfg["XcorrP"]["maxlag_factor"] ≈ 0.5
        @test cfg["XcorrP"]["trim"] ≈ [-2.0, 60.0]
        @test haskey(cfg, "XcorrS")
        @test cfg["XcorrS"]["trim"] ≈ [-2.0, 80.0]
        @test haskey(cfg, "Polarity")
        @test !haskey(cfg, "depth_indices")
        @test !haskey(cfg, "depth_vals")
        @test !haskey(cfg, "freq_low_idx")

        # Read paraspace
        ps = IO.read_paraspace(fn)
        @test haskey(ps, "strike")
        @test ps["strike"][1] ≈ 0.0
        @test ps["strike"][end] ≈ 355.0
        @test haskey(ps, "dip")
        @test ps["dip"][1] ≈ 0.0
        @test ps["dip"][end] ≈ 90.0
        @test haskey(ps, "rake")
        @test ps["rake"][1] ≈ -90.0
        @test ps["rake"][end] ≈ 90.0
        @test ps["depth"] ≈ [5.0, 10.0, 15.0]
        @test ps["frequency"] ≈ [0.05, 0.1, 0.5, 1.0]
    end

    @testset "paraspace round-trip" begin
        fn = tmpfile("test_paraspace.h5")
        HDF5.h5open(fn, "w") do f
        end  # create empty

        ps_data = Dict{String, Any}(
            "strike" => [0.0, 10.0, 20.0],
            "dip" => [0.0, 45.0, 90.0],
            "rake" => [-90.0, 0.0, 90.0],
            "depth" => [2.0, 8.0],
            "frequency" => [0.1, 0.5, 0.5, 2.0],
        )
        IO.write_paraspace(fn, ps_data)

        ps_read = IO.read_paraspace(fn)
        @test ps_read["strike"] ≈ [0.0, 10.0, 20.0]
        @test ps_read["dip"] ≈ [0.0, 45.0, 90.0]
        @test ps_read["rake"] ≈ [-90.0, 0.0, 90.0]
        @test ps_read["depth"] ≈ [2.0, 8.0]
        @test ps_read["frequency"] ≈ [0.1, 0.5, 0.5, 2.0]

        rm(fn; force = true)
    end

    @testset "output round-trip" begin
        ex = make_synthetic_output()
        fn = tmpfile("test_output.h5")

        h5o = HDF5.h5open(fn, "r") do f
            sol = f["solution"]
            @test read(sol, "strike") ≈ 130.0
            @test read(sol, "moment_tensor") ≈ [1.0, 2.0, 3.0, 0.5, -0.3, 1.2]

            unc = f["uncertainty"]
            @test read(unc, "strike_std") ≈ 5.0
            @test length(read(unc, "depth_range")) == 2

            per = f["per_phase"]
            @test length(read(per, "station_id")) == 3
            @test read(per, "selected") == Int32[1, 1, 0]

            sm = f["summary"]
            @test read(sm, "total_iterations") == 3
            @test String(read(sm, "convergence_reason")) == "user"
        end
    end

    @testset "h5create_group and h5exists" begin
        fn = tmpfile("test_helpers.h5")
        rm(fn; force = true)
        HDF5.h5open(fn, "w") do f
        end  # create empty

        IO.h5create_group(fn, "/a/b/c")
        @test IO.h5exists(fn, "/a")
        @test IO.h5exists(fn, "/a/b/c")
        @test IO.h5exists(fn, "/a/b")
        @test !IO.h5exists(fn, "/x/y/z")

        rm(fn)
    end

    @testset "NaN round-trip in matrix" begin
        fn = tmpfile("test_nan.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "misfits")
        end
        data = [NaN 1.0; 2.0 NaN]
        IO.write_misfits(fn, :nan_test, data)
        mis = IO.read_misfits(fn)
        @test size(mis[:nan_test]) == (2, 2)
        @test isnan(mis[:nan_test][1, 1])
        @test isnan(mis[:nan_test][2, 2])
    end
    @testset "recursive config read/write" begin
        # Test _write_group_recursive + read_config independently
        fn = tmpfile("test_deep_config.h5")
        deep_config = Dict{String, Any}(
            "basic" => "value",
            "level1" => Dict{String, Any}(
                "level2" => Dict("deep_key1" => 42.0, "deep_key2" => [1.0, 2.0, 3.0]),
                "shallow" => Int32(7),
            ),
        )
        HDF5.h5open(fn, "w") do f
            cg = HDF5.create_group(f, "config")
            IO._write_group_recursive(cg, deep_config)
        end

        cfg = IO.read_config(fn)
        @test cfg["basic"] == "value"
        @test cfg["level1"] isa Dict
        @test cfg["level1"]["level2"] isa Dict
        @test cfg["level1"]["level2"]["deep_key1"] ≈ 42.0
        @test cfg["level1"]["level2"]["deep_key2"] ≈ [1.0, 2.0, 3.0]
        @test cfg["level1"]["shallow"] == 7
        rm(fn; force = true)
    end

    @testset "write_strategy called twice (replacement)" begin
        fn = tmpfile("test_strategy_twice.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "strategy")
            HDF5.create_group(f, "trials")
            HDF5.create_group(f, "misfits")
        end
        strat = IO.Strategy(Int32[1, 2, 3], Int32[1, 3], Int32[2, 4], Int32(3))
        # Write first time
        IO.write_strategy(fn, strat)
        r1 = IO.read_strategy(fn)
        @test r1.depth_indices == Int32[1, 2, 3]
        @test r1.freq_low_idx == Int32[1, 3]
        @test r1.freq_high_idx == Int32[2, 4]
        @test r1.iteration == 3
        # Write second time (replacement)
        strat2 = IO.Strategy(Int32[3, 4], Int32[2], Int32[3], Int32(5))
        IO.write_strategy(fn, strat2)
        r2 = IO.read_strategy(fn)
        @test r2.depth_indices == Int32[3, 4]
        @test r2.freq_low_idx == Int32[2]
        @test r2.freq_high_idx == Int32[3]
        @test r2.iteration == 5
        rm(fn; force = true)
    end

    @testset "write_misfits called twice (replacement)" begin
        fn = tmpfile("test_misfits_twice.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "misfits")
        end
        data1 = reshape(Float64[1:6;], 2, 3)
        IO.write_misfits(fn, :xcorr, data1)
        # Verify first write
        mis1 = IO.read_misfits(fn)
        @test mis1[:xcorr] ≈ data1
        # Write second time — should replace not duplicate
        data2 = [7.0 8.0 9.0; 10.0 11.0 12.0]
        IO.write_misfits(fn, :xcorr, data2)
        mis2 = IO.read_misfits(fn)
        @test length(keys(mis2)) == 1  # only :xcorr, not duplicated
        @test mis2[:xcorr] ≈ data2    # second write values
        rm(fn; force = true)
    end

end

# Cleanup
rm.(filter(f -> endswith(f, ".h5"), readdir(@__DIR__; join = true)); force = true)
