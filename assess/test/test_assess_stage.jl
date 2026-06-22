#!/usr/bin/env julia

using HDF5IO
using HDF5
using Test

include(joinpath(@__DIR__, "..", "..", "shared", "AssessUtils.jl", "src", "AssessUtils.jl"))
using .AssessUtils: aggregate_misfits

include(joinpath(@__DIR__, "..", "src", "grid_refinement.jl"))
using .GridRefinement

# ═══════════════════════════════════════════════════════════
# Test: Assess Stage Integration
# ═══════════════════════════════════════════════════════════

const TEST_DIR = mktempdir()
const DATABASE_FILE = joinpath(TEST_DIR, "database.h5")
const STATUS_0_FILE = joinpath(TEST_DIR, "status_0.h5")
const STATUS_1_FILE = joinpath(TEST_DIR, "status_1.h5")
const ASSESS_PROJECT = normpath(joinpath(@__DIR__, ".."))
const ASSESS_SCRIPT = joinpath(ASSESS_PROJECT, "src", "assess.jl")

# ── Constants for fake data ──────────────────────────
const N_PHASES = 6    # 3 stations × 2 phases
const N_STATIONS = 3
const N_TRIALS = 27   # 3×3×3 SDR grid
const N_DEPTHS = 3
const N_FREQUENCIES = 1

# ── Best trial is #14 (middle of 27) ─────────────────
const BEST_TRIAL_IDX = 14
const BEST_SDR = [180.0, 45.0, 0.0]

println("Test directory: ", TEST_DIR)

# ═══════════════════════════════════════════════════════════
# 1. Create database.h5
# ═══════════════════════════════════════════════════════════
@testset "Setup database.h5" begin
    config = Dict{String, Any}(
        "depth_vals" => [5.0, 10.0, 15.0],
        "freq_bands_low" => [0.05],
        "freq_bands_high" => [0.2],
        "misfit_modules" => ["XCorr", "Polarity", "PSR"],
        "module_weights" => [1.0, 1.0, 1.0],
        "minimum_stations" => Int32(1),
        "freq_test_max_iter" => Int32(20),
    )
    greens = Dict{String, Dict{Int32, Matrix{Float64}}}()
    data = Dict{Int, Dict{Symbol, Dict{String, Any}}}()
    index = Index(
        String[],
        String[],
        Int32[],
        Float64[],
        Float64[],
        Matrix{Int32}(undef, 0, 0),
    )

    write_database(DATABASE_FILE, greens, data, index, config)
    @test isfile(DATABASE_FILE)

    # Verify config was written
    readback = read_config(DATABASE_FILE)
    @test get(readback, "depth_vals", nothing) == [5.0, 10.0, 15.0]
    @test get(readback, "freq_bands_low", nothing) == [0.05]
    @test length(get(readback, "misfit_modules", [])) == 3
    @test get(readback, "module_weights", nothing) == [1.0, 1.0, 1.0]
end

# ═══════════════════════════════════════════════════════════
# 2. Create status_0.h5 with strategy, trials, misfits
# ═══════════════════════════════════════════════════════════
@testset "Setup status_0.h5" begin
    # ── Strategy ──────────────────────────────────────
    strategy = Strategy(
        Float64(180.0), Float64(30.0), Int32(3),   # strike
        Float64(45.0),  Float64(15.0), Int32(3),   # dip
        Float64(0.0),   Float64(30.0), Int32(3),   # rake
        Int32[1, 2, 3],                            # depth_indices
        Int32[1],                                   # freq_indices
        ones(Int32, N_PHASES),                      # xcorr_phase_mask
        ones(Int32, N_STATIONS),                    # polarity_channel_mask
        ones(Int32, N_STATIONS),                    # psr_channel_mask
        [1.0, 1.0, 1.0],                           # module_weights
        [0.0, 0.0, 0.0],                           # best_sdr (unknown yet)
        Int32(1),                                   # best_depth_index
        0.0,                                        # best_misfit
        Int32(0),                                   # iteration
        Int32(0),                                   # converged
        "",                                         # convergence_reason
        zeros(Float64, N_FREQUENCIES, 3),           # freq_accumulated
        zeros(Float64, N_FREQUENCIES, 20),          # freq_misfit_curve
        Inf .* ones(Float64, N_DEPTHS),             # depth_misfit_accumulated
    )

    # ── Trials: 3×3×3 SDR grid ──────────────────────
    # strike: 150, 180, 210 (center 180)
    # dip: 30, 45, 60 (center 45)
    # rake: -30, 0, 30 (center 0)
    strikes = Float64[]
    dips    = Float64[]
    rakes   = Float64[]
    depths  = Float64[]
    depth_idxs = Int32[]
    freq_idxs  = Int32[]

    for s in [150.0, 180.0, 210.0]
        for d in [30.0, 45.0, 60.0]
            for r in [-30.0, 0.0, 30.0]
                push!(strikes, s)
                push!(dips, d)
                push!(rakes, r)
                push!(depths, 10.0)     # middle depth
                push!(depth_idxs, Int32(2))
                push!(freq_idxs, Int32(1))
            end
        end
    end

    trials = TrialSet(strikes, dips, rakes, depths, depth_idxs, freq_idxs)

    # ── Fake misfits: make trial #14 the winner ──────
    # xcorr: 6 phases × 27 trials
    xcorr = fill(0.5, N_PHASES, N_TRIALS)
    xcorr[:, BEST_TRIAL_IDX] .= 0.1  # best trial gets lower xcorr misfit

    # polarity: 3 stations × 27 trials
    polarity = fill(1.0, N_STATIONS, N_TRIALS)
    polarity[:, BEST_TRIAL_IDX] .= 0.0  # best trial matches all polarities

    # psr: 3 stations × 27 trials
    psr = fill(0.3, N_STATIONS, N_TRIALS)
    psr[:, BEST_TRIAL_IDX] .= 0.05  # best trial gets lower PSR misfit

    # ── Write status_0.h5 ────────────────────────────
    h5open(STATUS_0_FILE, "w") do f
    end

    write_strategy(STATUS_0_FILE, strategy)
    write_trials(STATUS_0_FILE, trials)
    write_misfits(STATUS_0_FILE, :xcorr, xcorr)
    write_misfits(STATUS_0_FILE, :polarity, polarity)
    write_misfits(STATUS_0_FILE, :psr, psr)

    @test isfile(STATUS_0_FILE)
    @test h5exists(STATUS_0_FILE, "strategy")
    @test h5exists(STATUS_0_FILE, "trials")
    @test h5exists(STATUS_0_FILE, "misfits/xcorr")
    @test h5exists(STATUS_0_FILE, "misfits/polarity")
    @test h5exists(STATUS_0_FILE, "misfits/psr")

    # Verify reading back
    ts = read_trials(STATUS_0_FILE)
    @test length(ts.strike) == N_TRIALS
    @test ts.strike[BEST_TRIAL_IDX] == BEST_SDR[1]
    @test ts.dip[BEST_TRIAL_IDX] == BEST_SDR[2]
    @test ts.rake[BEST_TRIAL_IDX] == BEST_SDR[3]
end

# ═══════════════════════════════════════════════════════════
# 3. Run assess.jl with "y" → continue
# ═══════════════════════════════════════════════════════════
@testset "assess.jl — continue (y)" begin
    rm(STATUS_1_FILE; force=true)
    open(pipeline(
        `echo "y"`,
        `julia --project=$ASSESS_PROJECT $ASSESS_SCRIPT $STATUS_0_FILE $DATABASE_FILE`
    )) do out
        output = read(out, String)
        @test contains(output, "status_1.h5")
        @test contains(output, "converged=0")
    end

    @test isfile(STATUS_1_FILE)

    # Read strategy from status_1.h5
    strat = read_strategy(STATUS_1_FILE)

    # converged=0 means continue
    @test strat.converged == Int32(0)

    # Step sizes should be halved
    @test strat.dstrike < 20.0
    @test abs(strat.dstrike - 15.0) < 1.0

    # Best SDR should match trial #14
    @test abs(strat.best_sdr[1] - BEST_SDR[1]) < 1.0
    @test abs(strat.best_sdr[2] - BEST_SDR[2]) < 1.0
    @test abs(strat.best_sdr[3] - BEST_SDR[3]) < 1.0

    # Iteration incremented
    @test strat.iteration == Int32(1)
end

# ═══════════════════════════════════════════════════════════
# 4. Run assess.jl with "N" → break
# ═══════════════════════════════════════════════════════════
@testset "assess.jl — break (N)" begin
    rm(STATUS_1_FILE; force=true)
    open(pipeline(
        `echo "N"`,
        `julia --project=$ASSESS_PROJECT $ASSESS_SCRIPT $STATUS_0_FILE $DATABASE_FILE`
    )) do out
        output = read(out, String)
        @test contains(output, "status_1.h5")
        @test contains(output, "converged=1")
    end

    @test isfile(STATUS_1_FILE)

    strat = read_strategy(STATUS_1_FILE)

    # converged=1 means break
    @test strat.converged == Int32(1)
    @test strat.convergence_reason == "user"
end

# ═══════════════════════════════════════════════════════════
# 5. Edge case: bad usage
# ═══════════════════════════════════════════════════════════
@testset "assess.jl — error handling" begin
    # No arguments
    proc = run(`julia --project=$ASSESS_PROJECT $ASSESS_SCRIPT`; wait=false)
    wait(proc)
    @test proc.exitcode != 0

    # Non-existent file
    proc = run(pipeline(`julia --project=$ASSESS_PROJECT $ASSESS_SCRIPT /nonexistent /also_fake`; stderr="/dev/null"); wait=false)
    wait(proc)
    @test proc.exitcode != 0
end

# ═══════════════════════════════════════════════════════════
# 6. Cleanup
# ═══════════════════════════════════════════════════════════
@testset "Cleanup" begin
    rm(TEST_DIR; recursive=true, force=true)
    @test !isdir(TEST_DIR)
end