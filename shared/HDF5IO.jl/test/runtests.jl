using Test, HDF5, HDF5IO

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
        write(gr, "component", ["E", "N", "E"])
        write(gr, "latitude", [32.0, 32.0, 32.5])
        write(gr, "longitude", [118.0, 118.0, 117.5])
        write(gr, "elevation", [150.0, 150.0, 200.0])
        write(gr, "dt", [0.01, 0.01, 0.02])
        write(gr, "begin_time", ["2024-03-15T08:21:00", "2024-03-15T08:21:00", "2024-03-15T08:21:00"])
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
    strategy = HDF5IO.Strategy(
        120.0, 10.0, 5,   # strike
        45.0, 5.0, 3,     # dip
        -90.0, 20.0, 4,   # rake
        Int32[0, 1, 2],   # depth_indices
        Int32[0, 3],       # freq_indices
        Int32[1, 1, 1, 0, 0, 1],  # xcorr_phase_mask
        Int32[1, 1, 0],    # polarity_station_mask
        Int32[1, 0, 1],    # psr_station_mask
        [0.5, 0.3, 0.2],  # module_weights
        [130.0, 50.0, -80.0],  # best_sdr
        Int32(2),           # best_depth_index
        0.023,              # best_misfit
        Int32(3),           # iteration
        Int32(0),           # converged
        "running",          # convergence_reason
        [1.0 2.0 3.0; 4.0 5.0 NaN],  # freq_accumulated
        [0.1 0.2; 0.3 0.4],           # freq_misfit_curve
        [0.05, 0.04, NaN, 0.02, 0.03],  # depth_misfit_accumulated
    )

    # ---- Trials ----
    trials = HDF5IO.TrialSet(
        collect(100.0:10.0:200.0),  # strike
        collect(40.0:5.0:60.0),     # dip
        collect(-120.0:20.0:-60.0), # rake
        fill(12.3, 11),             # depth
        Int32.(0:10),               # depth_idx
        Int32.(zeros(Int32, 11)),   # freq_idx
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
    HDF5IO.write_strategy(fn, strategy)
    HDF5IO.write_trials(fn, trials)
    HDF5IO.write_misfits(fn, :xcorr, xcorr_misfit)
    HDF5IO.write_misfits(fn, :polarity, polarity_misfit)
    HDF5IO.write_misfits(fn, :psr, psr_misfit)

    return (; strategy, trials, xcorr_misfit, polarity_misfit, psr_misfit)
end

function make_synthetic_database()
    fn = tmpfile("test_database.h5")

    green_phase = "NET.STA1.BHE.P"
    index = HDF5IO.Index(
        [green_phase, "NET.STA1.BHN.S", "NET.STA2.BHE.P"],
        ["P", "S", "P"],
        Int32[0, 0, 1],
        [120.0, 120.0, 125.0],
        [45.0, 45.0, 60.0],
        Int32[0 1; 0 2; 0 3],
    )

    greens = Dict(green_phase => Dict(Int32(0) => rand(100, 6), Int32(1) => rand(100, 6)))

    data = Dict(
        0 => Dict(
            :XCorr => Dict(
                green_phase => Dict(
                    "obs" => rand(100),
                    "gf" => rand(100, 6),
                    "synamp" => rand(6, 6),
                ),
            ),
        ),
    )

    config = Dict{String,Any}(
        "misfit_modules" => ["XCorr"],
        "module_weights" => [1.0],
        "depth_vals" => [5.0, 10.0, 15.0],
        "freq_bands_low" => [0.05, 0.1],
        "freq_bands_high" => [0.5, 1.0],
        "minimum_stations" => Int32(3),
        "freq_test_max_iter" => Int32(20),
        "xcorr" => Dict("maxlag_factor" => 0.5, "filter_order" => Int32(4),
                         "P_trim" => [-2.0, 60.0], "S_trim" => [-2.0, 80.0],
                         "select_threshold" => 0.7, "deselect_threshold" => 0.5),
    )

    HDF5IO.write_database(fn, greens, data, index, config)
    return (; greens, data, index, config)
end

function make_synthetic_output()
    fn = tmpfile("test_output.h5")

    solution = Dict(
        "strike" => 130.0, "dip" => 50.0, "rake" => -80.0,
        "depth" => 12.3, "moment_tensor" => [1.0, 2.0, 3.0, 0.5, -0.3, 1.2],
        "misfit" => 0.023,
    )

    uncertainty = Dict(
        "strike_std" => 5.0, "dip_std" => 3.0, "rake_std" => 7.0,
        "depth_range" => [10.0, 15.0],
        "freq_test_misfit_curve" => rand(2, 3),
    )

    per_station = Dict(
        "station_id" => ["NET.STA1", "NET.STA2", "NET.STA3"],
        "phase_type" => ["P", "P", "S"],
        "misfit_per_module" => rand(3, 3),
        "selected" => Int32[1, 1, 0],
        "cross_correlation" => [0.85, 0.72, 0.0],
    )

    summary = Dict(
        "total_iterations" => Int32(3),
        "total_trials" => Int32(1500),
        "convergence_reason" => "user",
    )

    HDF5IO.write_output(fn, solution, uncertainty, per_station, summary)
    return (; solution, uncertainty, per_station, summary)
end

# ─────────────────────────────
# Tests
# ─────────────────────────────

@testset "HDF5IO" begin
    @testset "read_event" begin
        make_synthetic_event()
        evt = HDF5IO.read_event(tmpfile("test_event.h5"))
        @test evt.longitude ≈ 118.5
        @test evt.latitude ≈ 32.1
        @test evt.depth ≈ 12.3
        @test evt.magnitude ≈ 4.7
        @test evt.origintime == "2024-03-15T08:22:00"
    end

    @testset "read_phase_picks" begin
        make_synthetic_phase_picks()
        picks = HDF5IO.read_phase_picks(tmpfile("test_phase_picks.h5"))
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
        stas = HDF5IO.read_stations(tmpfile("test_stations.h5"))
        @test length(stas) == 3
        @test stas[1].id == "NET.STA1.BHE.P"
        @test stas[1].latitude ≈ 32.0
        @test stas[1].dt ≈ 0.01
        @test stas[3].elevation ≈ 200.0
    end

    @testset "read_waveform" begin
        make_synthetic_waveforms()
        wf = HDF5IO.read_waveform(tmpfile("test_waveforms.h5"), "NET.STA1.BHE.P")
        @test wf ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
        wf2 = HDF5IO.read_waveform(tmpfile("test_waveforms.h5"), "NET.STA1.BHN.S")
        @test wf2 ≈ [0.5, 1.5, 2.5]
    end

    @testset "strategy round-trip with NaN" begin
        expected = make_synthetic_status()
        fn = tmpfile("test_status.h5")

        # Read strategy
        strat = HDF5IO.read_strategy(fn)
        @test strat.strike0 ≈ 120.0
        @test strat.nstrike == 5
        @test strat.depth_indices == Int32[0, 1, 2]
        @test strat.freq_indices == Int32[0, 3]
        @test strat.module_weights ≈ [0.5, 0.3, 0.2]
        @test strat.best_sdr ≈ [130.0, 50.0, -80.0]
        @test strat.best_depth_index == 2
        @test strat.best_misfit ≈ 0.023
        @test strat.iteration == 3
        @test strat.converged == 0
        @test strat.convergence_reason == "running"

        # NaN preservation in freq_accumulated
        @test size(strat.freq_accumulated) == (2, 3)
        @test strat.freq_accumulated[1, 1] ≈ 1.0
        @test isnan(strat.freq_accumulated[2, 3])

        # NaN in depth_misfit_accumulated
        @test length(strat.depth_misfit_accumulated) == 5
        @test isnan(strat.depth_misfit_accumulated[3])

        # freq_misfit_curve
        @test size(strat.freq_misfit_curve) == (2, 2)
        @test strat.freq_misfit_curve[1, 1] ≈ 0.1
    end

    @testset "trials round-trip" begin
        fn = tmpfile("test_status.h5")
        trials = HDF5IO.read_trials(fn)
        @test length(trials.strike) == 11
        @test trials.strike[1] ≈ 100.0
        @test trials.strike[end] ≈ 200.0
        @test trials.depth_idx isa Vector{Int32}
        @test all(trials.freq_idx .== 0)
    end

    @testset "misfits round-trip" begin
        fn = tmpfile("test_status.h5")
        mis = HDF5IO.read_misfits(fn)
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

        # Read index
        idx = HDF5IO.read_index(fn)
        @test idx.phase_ids == ex.index.phase_ids
        @test idx.phase_type == ex.index.phase_type
        @test idx.station_idx == ex.index.station_idx
        @test idx.distance ≈ ex.index.distance
        @test idx.azimuth ≈ ex.index.azimuth
        @test idx.greens_depth_idx == ex.index.greens_depth_idx

        # Read greens
        g = HDF5IO.read_greens(fn, "NET.STA1.BHE.P", Int32(0))
        @test size(g) == (100, 6)
        @test g ≈ ex.greens["NET.STA1.BHE.P"][Int32(0)]

        # Read config
        cfg = HDF5IO.read_config(fn)
        @test cfg["misfit_modules"] == ["XCorr"]
        @test cfg["depth_vals"] ≈ [5.0, 10.0, 15.0]
        @test cfg["minimum_stations"] == 3
        @test haskey(cfg, "xcorr")
        @test cfg["xcorr"]["maxlag_factor"] ≈ 0.5
        @test cfg["xcorr"]["P_trim"] ≈ [-2.0, 60.0]
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

            per = f["per_station"]
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
        HDF5.h5open(fn, "w") do f end  # create empty

        HDF5IO.h5create_group(fn, "/a/b/c")
        @test HDF5IO.h5exists(fn, "/a")
        @test HDF5IO.h5exists(fn, "/a/b/c")
        @test HDF5IO.h5exists(fn, "/a/b")
        @test !HDF5IO.h5exists(fn, "/x/y/z")

        rm(fn)
    end

    @testset "NaN round-trip in matrix" begin
        fn = tmpfile("test_nan.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "misfits")
        end
        data = [NaN 1.0; 2.0 NaN]
        HDF5IO.write_misfits(fn, :nan_test, data)
        mis = HDF5IO.read_misfits(fn)
        @test size(mis[:nan_test]) == (2, 2)
        @test isnan(mis[:nan_test][1, 1])
        @test isnan(mis[:nan_test][2, 2])
    end
@testset "recursive config read/write" begin
        # Create a database with deep nested config
        fn = tmpfile("test_deep_config.h5")
        deep_config = Dict{String,Any}(
            "basic" => "value",
            "level1" => Dict{String,Any}(
                "level2" => Dict(
                    "deep_key1" => 42.0,
                    "deep_key2" => [1.0, 2.0, 3.0],
                ),
                "shallow" => Int32(7),
            ),
        )
        # Build minimal greens/data/index for write_database
        greens = Dict{String, Dict{Int32, Matrix{Float64}}}()
        data = Dict{Int, Dict{Symbol, Dict{String, Any}}}()
        index = HDF5IO.Index(
            String[], String[], Int32[], Float64[], Float64[], Int32[0;;]
        )
        HDF5IO.write_database(fn, greens, data, index, deep_config)

        cfg = HDF5IO.read_config(fn)
        @test cfg["basic"] == "value"
        @test cfg["level1"] isa Dict
        @test cfg["level1"]["level2"] isa Dict
        @test cfg["level1"]["level2"]["deep_key1"] ≈ 42.0
        @test cfg["level1"]["level2"]["deep_key2"] ≈ [1.0, 2.0, 3.0]
        @test cfg["level1"]["shallow"] == 7
        rm(fn; force=true)
    end

    @testset "write_strategy called twice (replacement)" begin
        fn = tmpfile("test_strategy_twice.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "strategy")
            HDF5.create_group(f, "trials")
            HDF5.create_group(f, "misfits")
        end
        strat = HDF5IO.Strategy(
            120.0, 10.0, 5, 45.0, 5.0, 3, -90.0, 20.0, 4,
            Int32[0, 1, 2], Int32[0, 3],
            Int32[1, 1, 1], Int32[1, 1], Int32[1, 1],
            [0.5, 0.3, 0.2], [130.0, 50.0, -80.0],
            Int32(2), 0.023, Int32(3), Int32(0), "running",
            [1.0 2.0; 3.0 NaN], [0.1 0.2; 0.3 0.4], [0.05, NaN],
        )
        # Write first time
        HDF5IO.write_strategy(fn, strat)
        r1 = HDF5IO.read_strategy(fn)
        @test r1.strike0 ≈ 120.0
        @test r1.nstrike == 5
        # Write second time (replacement)
        strat2 = HDF5IO.Strategy(
            200.0, 5.0, 10, 60.0, 3.0, 2, 0.0, 15.0, 6,
            Int32[3, 4], Int32[1],
            Int32[0, 1], Int32[0], Int32[0],
            [0.6, 0.2, 0.2], [200.0, 60.0, 0.0],
            Int32(3), 0.015, Int32(5), Int32(1), "converged",
            [5.0;;], [0.05;;], [0.01, 0.02],
        )
        HDF5IO.write_strategy(fn, strat2)
        r2 = HDF5IO.read_strategy(fn)
        @test r2.strike0 ≈ 200.0
        @test r2.nstrike == 10
        @test r2.converged == 1
        @test r2.convergence_reason == "converged"
        rm(fn; force=true)
    end

    @testset "write_misfits called twice (replacement)" begin
        fn = tmpfile("test_misfits_twice.h5")
        HDF5.h5open(fn, "w") do f
            HDF5.create_group(f, "misfits")
        end
        data1 = reshape(Float64[1:6;], 2, 3)
        HDF5IO.write_misfits(fn, :xcorr, data1)
        # Verify first write
        mis1 = HDF5IO.read_misfits(fn)
        @test mis1[:xcorr] ≈ data1
        # Write second time — should replace not duplicate
        data2 = [7.0 8.0 9.0; 10.0 11.0 12.0]
        HDF5IO.write_misfits(fn, :xcorr, data2)
        mis2 = HDF5IO.read_misfits(fn)
        @test length(keys(mis2)) == 1  # only :xcorr, not duplicated
        @test mis2[:xcorr] ≈ data2    # second write values
        rm(fn; force=true)
    end

end

# Cleanup
rm.(filter(f -> endswith(f, ".h5"), readdir(@__DIR__; join = true)); force = true)