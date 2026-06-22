#!/usr/bin/env julia
#=
Test: Preprocess stage integration (T17)

Creates a status_0.h5 with strategy, runs the preprocess stage,
and verifies /trials are created with correct count while /strategy is unchanged.

Usage:
  julia --project=preprocess preprocess/test/test_preprocess_stage.jl
=#

using Test
using HDF5

# ─────────────────────────────────────────────────────────
# Include modules
# ─────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "..", "shared", "HDF5IO.jl", "src", "HDF5IO.jl"))
using .HDF5IO

include(joinpath(@__DIR__, "..", "..", "shared", "TrialGen.jl", "src", "TrialGen.jl"))
using .TrialGen

# ─────────────────────────────────────────────────────────
# Helper: create a status file with a strategy
# ─────────────────────────────────────────────────────────

function create_status_file(status_path::String, strategy::HDF5IO.Strategy)
    # Create the file and write strategy
    h5open(status_path, "w") do f
        # nothing to do — write_strategy handles it
    end
    HDF5IO.write_strategy(status_path, strategy)
end

# ─────────────────────────────────────────────────────────
# Helper: read a strategy from raw HDF5 (pre-read_strategy)
# ─────────────────────────────────────────────────────────

function read_strategy_raw(status_path::String)
    return HDF5IO.read_strategy(status_path)
end

# ─────────────────────────────────────────────────────────
# Helper: run preprocess core logic
# ─────────────────────────────────────────────────────────

function run_preprocess(status_file::String, database_file::Union{String,Nothing})
    strategy = HDF5IO.read_strategy(status_file)

    grid = TrialGen.GridStrategy(
        strategy.strike0, strategy.dstrike, strategy.nstrike,
        strategy.dip0, strategy.ddip, strategy.ndip,
        strategy.rake0, strategy.drake, strategy.nrake,
        strategy.depth_indices, strategy.freq_indices,
        strategy.best_depth_index,
    )

    if database_file !== nothing && HDF5IO.h5exists(database_file, "config/depth_vals")
        depth_vals = h5open(database_file, "r") do f
            read(f["config/depth_vals"])
        end
    else
        max_idx = isempty(grid.depth_indices) ? grid.best_depth_index : maximum(grid.depth_indices)
        depth_vals = collect(Float64, 1:max(max_idx, 1))
    end

    trialgen_trials = TrialGen.generate_trials(grid, depth_vals)

    hdf5_trials = HDF5IO.TrialSet(
        trialgen_trials.strike, trialgen_trials.dip, trialgen_trials.rake,
        trialgen_trials.depth, trialgen_trials.depth_idx, trialgen_trials.freq_idx,
    )
    HDF5IO.write_trials(status_file, hdf5_trials)

    return length(trialgen_trials.strike)
end

# ─────────────────────────────────────────────────────────
# Helper: count trials from the grid params
# ─────────────────────────────────────────────────────────

function expected_trial_count(s::HDF5IO.Strategy)
    n_s = max(s.nstrike, 1)
    n_d = max(s.ndip, 1)
    n_r = max(s.nrake, 1)
    n_depth = max(length(s.depth_indices), 1)
    n_freq = max(length(s.freq_indices), 1)
    return Int(n_s * n_d * n_r * n_depth * n_freq)
end

# ─────────────────────────────────────────────────────────
# Helper: check /trials exists and has correct fields
# ─────────────────────────────────────────────────────────

function check_trials_group(status_path::String, expected_count::Int)
    h5open(status_path, "r") do f
        @test haskey(f, "trials")
        gr = f["trials"]
        @test haskey(gr, "strike")
        @test haskey(gr, "dip")
        @test haskey(gr, "rake")
        @test haskey(gr, "depth")
        @test haskey(gr, "depth_idx")
        @test haskey(gr, "freq_idx")

        strikes = read(gr["strike"])
        @test length(strikes) == expected_count
        @test length(read(gr["dip"])) == expected_count
        @test length(read(gr["rake"])) == expected_count
        @test length(read(gr["depth"])) == expected_count
        @test length(read(gr["depth_idx"])) == expected_count
        @test length(read(gr["freq_idx"])) == expected_count
    end
end

# ─────────────────────────────────────────────────────────
# Helper: verfy strategy unchanged
# ─────────────────────────────────────────────────────────

function check_strategy_unchanged(status_path::String, original::HDF5IO.Strategy)
    after = HDF5IO.read_strategy(status_path)
    @test after.strike0 ≈ original.strike0
    @test after.dstrike ≈ original.dstrike
    @test after.nstrike == original.nstrike
    @test after.dip0 ≈ original.dip0
    @test after.ddip ≈ original.ddip
    @test after.ndip == original.ndip
    @test after.rake0 ≈ original.rake0
    @test after.drake ≈ original.drake
    @test after.nrake == original.nrake
    @test after.depth_indices == original.depth_indices
    @test after.freq_indices == original.freq_indices
    @test after.best_depth_index == original.best_depth_index
    @test after.iteration == original.iteration
    @test after.converged == original.converged
end

# ─────────────────────────────────────────────────────────
# Test 1: All axes varying (3×3×3 SDR × 2 depths × 2 freqs)
# ─────────────────────────────────────────────────────────

@testset "3×3×3 SDR × 2 depths × 2 freqs" begin
    status_path = "/tmp/test_preprocess_stage_1.h5"
    rm(status_path; force=true)

    strategy = HDF5IO.Strategy(
        45.0, 10.0, Int32(3),     # strike: 45, 55, 65
        20.0,  5.0, Int32(3),     # dip: 20, 25, 30
        -90.0, 5.0, Int32(3),     # rake: -90, -85, -80
        Int32[1, 2],              # 2 depth indices
        Int32[1, 2],              # 2 freq indices
        Int32[], Int32[], Int32[], # masks (empty for now)
        [1.0, 0.5, 0.3],          # module weights
        [0.0, 0.0, 0.0],          # best_sdr
        Int32(1),                  # best_depth_index
        0.0,                       # best_misfit
        Int32(0),                  # iteration
        Int32(0),                  # converged
        "",
        zeros(0, 3),               # freq_accumulated
        zeros(0, 0),               # freq_misfit_curve
        Float64[],                  # depth_misfit_accumulated
    )

    expected = expected_trial_count(strategy)  # 3×3×3×2×2 = 108

    create_status_file(status_path, strategy)
    @test isfile(status_path)

    nt = run_preprocess(status_path, nothing)
    @test nt == expected

    check_trials_group(status_path, expected)
    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
end

# ─────────────────────────────────────────────────────────
# Test 2: Fixed SDR, varying depths only
# ─────────────────────────────────────────────────────────

@testset "Fixed SDR, varying depths (nstrike=0, ndip=0, nrake=0)" begin
    status_path = "/tmp/test_preprocess_stage_2.h5"
    rm(status_path; force=true)

    strategy = HDF5IO.Strategy(
        45.0, 10.0, Int32(0),     # strike: fixed at 45
        30.0,  5.0, Int32(0),     # dip: fixed at 30
        -90.0, 5.0, Int32(0),     # rake: fixed at -90
        Int32[1, 2, 3],           # 3 depth indices
        Int32[1],                  # 1 freq
        Int32[], Int32[], Int32[],
        [1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0],
        Int32(1),
        0.0,
        Int32(0),
        Int32(0),
        "",
        zeros(0, 3),
        zeros(0, 0),
        Float64[],
    )

    expected = expected_trial_count(strategy)  # 1×1×1×3×1 = 3

    create_status_file(status_path, strategy)
    nt = run_preprocess(status_path, nothing)
    @test nt == expected

    check_trials_group(status_path, expected)
    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
end

# ─────────────────────────────────────────────────────────
# Test 3: Single trial (all fixed)
# ─────────────────────────────────────────────────────────

@testset "Single trial — all axes fixed" begin
    status_path = "/tmp/test_preprocess_stage_3.h5"
    rm(status_path; force=true)

    strategy = HDF5IO.Strategy(
        45.0, 10.0, Int32(0),
        30.0,  5.0, Int32(0),
        -90.0, 5.0, Int32(0),
        Int32[],                  # empty → use best_depth_index
        Int32[],                  # empty → use freq 1
        Int32[], Int32[], Int32[],
        [1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0],
        Int32(2),                 # best_depth_index
        0.0,
        Int32(0),
        Int32(0),
        "",
        zeros(0, 3),
        zeros(0, 0),
        Float64[],
    )

    expected = 1  # single trial

    create_status_file(status_path, strategy)
    nt = run_preprocess(status_path, nothing)
    @test nt == expected

    check_trials_group(status_path, expected)

    # Check trial values
    h5open(status_path, "r") do f
        gr = f["trials"]
        @test read(gr["strike"])[1] ≈ 45.0
        @test read(gr["dip"])[1] ≈ 30.0
        @test read(gr["rake"])[1] ≈ -90.0
        @test read(gr["depth_idx"])[1] == Int32(2)
        @test read(gr["freq_idx"])[1] == Int32(1)
    end

    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
end

# ─────────────────────────────────────────────────────────
# Test 4: With database.h5 for depth_vals
# ─────────────────────────────────────────────────────────

@testset "With database.h5 depth_vals lookup" begin
    status_path = "/tmp/test_preprocess_stage_4.h5"
    db_path = "/tmp/test_preprocess_database_4.h5"
    rm(status_path; force=true)
    rm(db_path; force=true)

    # Create a minimal database.h5 with depth_vals
    h5open(db_path, "w") do f
        gr = HDF5.create_group(f, "config")
        write(gr, "depth_vals", [5.0, 10.0, 15.0, 20.0, 25.0])
    end

    strategy = HDF5IO.Strategy(
        45.0, 0.0, Int32(0),
        30.0, 0.0, Int32(0),
        -90.0, 0.0, Int32(0),
        Int32[1, 3, 5],           # depth indices 1, 3, 5
        Int32[1],
        Int32[], Int32[], Int32[],
        [1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0],
        Int32(1),
        0.0,
        Int32(0),
        Int32(0),
        "",
        zeros(0, 3),
        zeros(0, 0),
        Float64[],
    )

    expected = 3  # 1×1×1×3×1 = 3

    create_status_file(status_path, strategy)
    nt = run_preprocess(status_path, db_path)
    @test nt == expected

    # Verify depth values came from database
    h5open(status_path, "r") do f
        gr = f["trials"]
        depths = read(gr["depth"])
        @test depths ≈ [5.0, 15.0, 25.0]
        depth_idxs = read(gr["depth_idx"])
        @test depth_idxs == Int32[1, 3, 5]
    end

    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
    rm(db_path; force=true)
end

# ─────────────────────────────────────────────────────────
# Test 5: Large grid — 5×5×5 SDR × 5 depths × 3 freqs
# ─────────────────────────────────────────────────────────

@testset "Large grid — 5×5×5 × 5 × 3 = 1875 trials" begin
    status_path = "/tmp/test_preprocess_stage_5.h5"
    rm(status_path; force=true)

    strategy = HDF5IO.Strategy(
        0.0, 22.5, Int32(5),     # 0, 22.5, 45, 67.5, 90
        0.0, 10.0, Int32(5),     # 0, 10, 20, 30, 40
        -90.0, 9.0, Int32(5),    # -90, -81, -72, -63, -54
        Int32[1, 2, 3, 4, 5],    # 5 depths
        Int32[1, 2, 3],           # 3 freqs
        Int32[], Int32[], Int32[],
        [1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0],
        Int32(1),
        0.0,
        Int32(0),
        Int32(0),
        "",
        zeros(0, 3),
        zeros(0, 0),
        Float64[],
    )

    expected = 5 * 5 * 5 * 5 * 3  # 1875

    create_status_file(status_path, strategy)
    nt = run_preprocess(status_path, nothing)
    @test nt == expected

    check_trials_group(status_path, expected)

    # Spot-check trial ordering: first trial should be min on all axes
    h5open(status_path, "r") do f
        gr = f["trials"]
        strikes = read(gr["strike"])
        dips = read(gr["dip"])
        rakes = read(gr["rake"])

        # First trial: strike=0, dip=0, rake=-90
        @test strikes[1] ≈ 0.0
        @test dips[1] ≈ 0.0
        @test rakes[1] ≈ -90.0

        # Last trial: strike=90, dip=40, rake=-54
        @test strikes[end] ≈ 90.0
        @test dips[end] ≈ 40.0
        @test rakes[end] ≈ -54.0
    end

    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
end

# ─────────────────────────────────────────────────────────
# Test 6: Mixed varying/fixed axes
# ─────────────────────────────────────────────────────────

@testset "Varying strike/dip, fixed rake, empty freq_indices, empty depth_indices" begin
    status_path = "/tmp/test_preprocess_stage_6.h5"
    rm(status_path; force=true)

    strategy = HDF5IO.Strategy(
        30.0, 15.0, Int32(3),     # 3 strikes: 30, 45, 60
        20.0, 10.0, Int32(2),     # 2 dips: 20, 30
        90.0,  0.0, Int32(0),     # fixed rake: 90
        Int32[],                   # empty → best_depth_index
        Int32[],                   # empty → freq 1
        Int32[], Int32[], Int32[],
        [1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0],
        Int32(3),
        0.0,
        Int32(0),
        Int32(0),
        "",
        zeros(0, 3),
        zeros(0, 0),
        Float64[],
    )

    expected = 3 * 2 * 1 * 1 * 1  # 6

    create_status_file(status_path, strategy)
    nt = run_preprocess(status_path, nothing)
    @test nt == expected

    check_trials_group(status_path, expected)

    # Verify ordering: strike outermost, dip next
    h5open(status_path, "r") do f
        gr = f["trials"]
        strikes = read(gr["strike"])
        dips = read(gr["dip"])
        rakes = read(gr["rake"])

        # strike varies first: [30,30,45,45,60,60] (each strike × each dip)
        @test strikes[1:2] ≈ [30.0, 30.0]
        @test strikes[3:4] ≈ [45.0, 45.0]
        @test strikes[5:6] ≈ [60.0, 60.0]

        # dip varies second: [20,30,20,30,20,30]
        @test dips[1] ≈ 20.0
        @test dips[2] ≈ 30.0
        @test dips[3] ≈ 20.0

        # rake fixed
        @test all(r -> r ≈ 90.0, rakes)
    end

    check_strategy_unchanged(status_path, strategy)

    rm(status_path; force=true)
end

println("\nAll preprocess stage integration tests passed!")