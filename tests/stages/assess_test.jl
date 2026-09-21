# assess_test.jl — Stage 4 (assess) tests.
#
# Chains input → forward on a small trial set, then runs assess.jl and checks:
#   - `/misfits/XcorrP|S` == 1 − cc_max (extract correct)
#   - misfit matrix shape [N_entries × N_trials]
#   - channel indices and station-level matrices are written independently
#   - empty `.decision.txt` = converged when DATA_DIR set
#   - `/strategy` and `/trials` untouched
#
# Usage:
#   julia --project=. tests/stages/assess_test.jl

using Test
using HDF5

using IO, Search

include("test_util.jl")

@testset "assess stage" begin
    mktempdir() do dir
        make_synthetic(dir; nsta = 3)
        config = write_test_config(dir)
        open(config, "a") do io
            println(
                io,
                "Config.@objective CombinedP = abs(1 - maxCC(p_observed, p_synthetic; maxlag = 3)) + 0.01 * abs(lagCC(p_observed, p_synthetic; maxlag = 3))",
            )
        end

        r = run_stage_script(joinpath("scripts", "input.jl"), [config])
        @test r.ok

        db = joinpath(dir, "database.h5")
        status_dir = joinpath(dir, "status")
        mkpath(status_dir)
        status0 = joinpath(status_dir, "status_0.h5")
        mv(joinpath(dir, "status_0.h5"), status0)

        # Small trial set (108 trials) covering all three STF durations.
        strat = IO.Strategy(
            0.0,
            5.0,
            Int32(3),
            0.0,
            5.0,
            Int32(2),
            -90.0,
            5.0,
            Int32(2),
            Int32[1, 2, 3],
            Int32[1],
            Int32[1, 2, 3],
            Int32(0),
        )
        trials = Search.generate_trials(strat)
        IO.write_trials(status0, trials)
        N_trials = length(trials.strike_idx)

        fwd = run_forward(db, status0)
        @test fwd.ok

        # ── Assess with DATA_DIR set (exercises the decision-file path) ──
        a = run_stage_script(
            joinpath("scripts", "assess.jl"),
            [db, status0];
            env = Dict("DATA_DIR" => dir),
        )
        @testset "run" begin
            @test a.ok
            @test isfile(joinpath(dir, ".decision.txt"))
            @test isempty(read(joinpath(dir, ".decision.txt"), String))  # converged
        end

        @testset "misfits written" begin
            h5open(status0, "r") do f
                @test haskey(f, "/misfits")
                for name in ("XcorrP", "XcorrS")
                    @test haskey(f["/misfits"], name)
                    m = read(f["/misfits/$name"])
                    n_ent = size(m, 1)  # entries; channel_id may be absent when 0 entries
                    @test size(m) == (n_ent, N_trials)
                    @test all(0.0 .<= m .<= 2.0)

                    # extract: misfit = 1 − cc_max
                    cc = read(f["/intermediates/$name/cc_max"])  # (N_trials, N_entries)
                    expect = 1.0 .- permutedims(cc)
                    @test m ≈ expect atol = 1e-12
                end
                for name in ("LagP", "LagS")
                    @test haskey(f["/misfits"], name)
                    m = read(f["/misfits/$name"])
                    @test size(m, 2) == N_trials
                    @test all(isfinite, m)
                end
                for name in ("Psr", "PolarityP")
                    @test haskey(f["/misfits"], name)
                    m = read(f["/misfits/$name"])
                    @test size(m, 2) == N_trials
                    @test all(m .>= 0.0)
                end
                combined = read(f["/misfits/CombinedP"])
                cc = permutedims(read(f["/intermediates/XcorrP/cc_max"]))
                lag = permutedims(read(f["/intermediates/XcorrP/best_lag"])) .* 0.01
                @test combined ≈ abs.(1.0 .- cc) .+ 0.01 .* abs.(lag) atol = 1e-12
                @test haskey(f, "/intermediates/XcorrP/syn_energy")
                @test haskey(f, "/intermediates/XcorrS/syn_energy")
                @test haskey(f, "/intermediates/XcorrP/amp_scale")
                @test haskey(f, "/intermediates/XcorrP/sign_scale")
                @test haskey(f, "/aggregate/total")
                aggregate = read(f["/aggregate/total"])
                @test size(aggregate) == (N_trials,)
                @test all(aggregate .>= 0.0)
                aggregate_channels = String.(read(f["/aggregate/channel/channel_id"]))
                aggregate_station_idx = read(f["/aggregate/channel/station_idx"])
                channel_phase_sum = read(f["/aggregate/channel/phase_sum"])
                channel_direct_sum = read(f["/aggregate/channel/direct_sum"])
                channel_total = read(f["/aggregate/channel/total"])
                expected_phase = zeros(size(channel_total))
                expected_direct = zeros(size(channel_total))
                for name in ("XcorrP", "XcorrS", "LagP", "LagS", "Psr", "PolarityP", "CombinedP")
                    @test haskey(f, "/misfit_index/$name/channel_id")
                    ids = String.(read(f["/misfit_index/$name/channel_id"]))
                    values = read(f["/misfits/$name"])
                    @test length(ids) == size(values, 1)
                    name in ("LagP", "LagS") && (values = abs.(values))
                    target_matrix =
                        name in ("XcorrP", "XcorrS", "LagP", "LagS") ? expected_phase :
                        expected_direct
                    for (row, channel_id) in enumerate(ids)
                        target = only(findall(==(channel_id), aggregate_channels))
                        target_matrix[target, :] .+= values[row, :]
                    end
                end
                @test channel_phase_sum ≈ expected_phase
                @test channel_direct_sum ≈ expected_direct
                @test channel_total ≈ expected_phase .+ expected_direct

                station_channel = read(f["/aggregate/station/channel_sum"])
                expected_station = zeros(size(station_channel))
                for row in axes(channel_total, 1)
                    expected_station[aggregate_station_idx[row], :] .+= channel_total[row, :]
                end
                @test station_channel ≈ expected_station
                @test all(iszero, read(f["/aggregate/station/direct_sum"]))
                @test read(f["/aggregate/station/total"]) ≈ expected_station
                @test aggregate ≈ vec(sum(expected_station; dims = 1))
            end
            h5open(db, "r") do f
                @test string.(read(f["/config/misfit_modules"])) ==
                      ["XcorrP", "XcorrS", "LagP", "LagS", "Psr", "PolarityP", "CombinedP"]
            end
        end

        @testset "strategy & trials untouched" begin
            h5open(status0, "r") do f
                @test read(f["/strategy/iteration"]) == 0
                @test read(f["/strategy/nstrike"]) == 72  # input's full grid, untouched by assess
                @test length(read(f["/trials/strike_idx"])) == N_trials
            end
        end
    end
end
