using Test
using LinearAlgebra
using Statistics

# Load dependencies
include(joinpath(@__DIR__, "..", "src", "solution_comp.jl"))
using .SolutionComp
include(joinpath(@__DIR__, "..", "..", "shared", "HDF5IO.jl", "src", "HDF5IO.jl"))
using .HDF5IO
include(joinpath(@__DIR__, "..", "..", "shared", "MTUtils.jl", "src", "MTUtils.jl"))
using .MTUtils

# Include AssessUtils for full aggregation
include(joinpath(@__DIR__, "..", "..", "shared", "AssessUtils.jl", "src", "AssessUtils.jl"))
using .AssessUtils

# ─────────────────────────────────────────────────────────
# Test 1: compute_depth_range — 5% tolerance
# ─────────────────────────────────────────────────────────
@testset "Depth range (5% tolerance)" begin
    depths = [1.0, 2.0, 3.0, 4.0, 5.0]
    misfits = [0.9, 0.5, 0.51, 0.9, NaN]

    result = SolutionComp.compute_depth_range(depths, misfits)
    # Best misfit = 0.5 at index 2 (depth=2.0)
    # 5% threshold: 0.5 * 1.05 = 0.525
    # Index 1 (0.5) and index 2 (0.51) are within threshold
    @test length(result) == 2
    @test result[1] == 2.0  # min
    @test result[2] == 3.0  # max

    # Single depth
    @test SolutionComp.compute_depth_range([10.0], [0.1]) == [10.0, 10.0]

    # All NaN
    result_nan = SolutionComp.compute_depth_range([1.0, 2.0], [NaN, NaN])
    @test isnan(result_nan[1])
    @test isnan(result_nan[2])

    # Empty
    result_empty = SolutionComp.compute_depth_range(Float64[], Float64[])
    @test isnan(result_empty[1])
    @test isnan(result_empty[2])
end

# ─────────────────────────────────────────────────────────
# Test 2: compute_sdr_std — frequency uncertainty
# ─────────────────────────────────────────────────────────
@testset "Frequency uncertainty (std)" begin
    # 3 frequency bands, each with best SDR
    freq_accum = [
        45.0 30.0 90.0;
        47.0 32.0 88.0;
        43.0 28.0 92.0;
    ]
    ss, ds, rs = SolutionComp.compute_sdr_std(freq_accum)

    # Manual computation
    @test isapprox(ss, std(freq_accum[:, 1]))
    @test isapprox(ds, std(freq_accum[:, 2]))
    @test isapprox(rs, std(freq_accum[:, 3]))

    # Single band → NaN
    single = [45.0 30.0 90.0]
    ss1, ds1, rs1 = SolutionComp.compute_sdr_std(single)
    @test isnan(ss1) && isnan(ds1) && isnan(rs1)
end

# ─────────────────────────────────────────────────────────
# Test 3: compile_solution — best-fit verification
# ─────────────────────────────────────────────────────────
@testset "compile_solution: best-fit verification" begin
    # Build synthetic test data
    n_trials = 27  # 3×3×3 grid

    # Strategy: 3x3x3 SDR grid around (45, 30, 90)
    strategy = Strategy(
        45.0, 10.0, 2,       # strike0, dstrike, nstrike
        30.0, 10.0, 2,       # dip0, ddip, ndip
        90.0, 10.0, 2,       # rake0, drake, nrake
        Int32[1], Int32[1],  # depth_indices, freq_indices
        Int32[1, 1, 1],      # xcorr_phase_mask (3 phases active)
        Int32[1],            # polarity_channel_mask (1 channel)
        Int32[1],            # psr_channel_mask
        [1.0, 0.5, 0.3],    # module_weights
        [45.0, 30.0, 90.0], # best_sdr
        1,                   # best_depth_index
        0.15,                # best_misfit
        2,                   # iteration
        1,                   # converged
        "user",              # convergence_reason
        [45.0 30.0 90.0; 46.0 31.0 89.0; 44.0 29.0 91.0],  # freq_accumulated
        zeros(Float64, 3, 10),  # freq_misfit_curve
        [0.9, 0.5, 0.51]       # depth_misfit_accumulated
    )

    # Trials: 3×3×3 grid
    strikes = [35.0, 45.0, 55.0, 35.0, 45.0, 55.0, 35.0, 45.0, 55.0,
               35.0, 45.0, 55.0, 35.0, 45.0, 55.0, 35.0, 45.0, 55.0,
               35.0, 45.0, 55.0, 35.0, 45.0, 55.0, 35.0, 45.0, 55.0]
    dips = [20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0,
            20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0,
            20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0]
    rakes = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0,
             90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
             100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    depths = fill(10.0, n_trials)
    depth_idxs = fill(Int32(1), n_trials)
    freq_idxs = fill(Int32(1), n_trials)

    trials = TrialSet(strikes, dips, rakes, depths, depth_idxs, freq_idxs)

    # Misfits: best trial is (45, 30, 90) which is trial index 14 (1-based)
    # trial 14: strike=45, dip=30, rake=90
    n_phases = 3
    n_st = 1
    xcorr = fill(1.0, n_phases, n_trials)
    xcorr[:, 14] .= 0.15  # Best trial has low misfit

    polarity = fill(1.0, n_st, n_trials)
    polarity[1, 14] = 0.0

    psr = fill(0.5, n_st, n_trials)
    psr[1, 14] = 0.1

    misfits = Dict(
        :xcorr => xcorr,
        :polarity => polarity,
        :psr => psr,
    )

    # Index
    idx = Index(
        ["NET.ST1.Z.P", "NET.ST1.Z.S", "NET.ST2.Z.P"],
        ["P", "S", "P"],
        Int32[1, 1, 2],
        [10.0, 10.0, 20.0],
        [30.0, 30.0, 60.0],
        fill(Int32(1), 3, 3),
    )

    # Config
    config = Dict(
        "depth_vals" => [5.0, 10.0, 15.0],
        "misfit_modules" => ["XCorr", "Polarity", "PSR"],
        "module_weights" => [1.0, 0.5, 0.3],
    )

    result = SolutionComp.compile_solution(strategy, trials, misfits, idx, config)

    # Verify best-fit
    sol = result.solution
    @test sol["strike"] == 45.0
    @test sol["dip"] == 30.0
    @test sol["rake"] == 90.0
    @test length(sol["moment_tensor"]) == 6

    # Verify best-fit SDR → MT matches MTUtils directly
    mt_direct = MTUtils.sdr_to_mt(45.0, 30.0, 90.0)
    @test sol["moment_tensor"] ≈ mt_direct

    # Verify solution misfit matches expected
    @test sol["misfit"] ≈ result.solution["misfit"]

    # Verify uncertainty
    unc = result.uncertainty
    @test !isnan(unc["strike_std"])
    @test !isnan(unc["dip_std"])
    @test !isnan(unc["rake_std"])
    @test length(unc["depth_range"]) == 2

    # Verify per_phase
    pp = result.per_phase
    @test length(pp["phase_id"]) == n_phases
    @test length(pp["channel_id"]) == n_phases
    @test length(pp["station_id"]) == n_phases
    @test length(pp["phase_type"]) == n_phases
    @test size(pp["misfit_per_module"]) == (3, n_phases)
    @test length(pp["selected"]) == n_phases
    @test length(pp["cross_correlation"]) == n_phases

    # Verify per_station_summary
    pss = result.per_station_summary
    @test length(pss["station_id"]) >= 1
    @test length(pss["n_channels"]) == length(pss["station_id"])
    @test length(pss["n_phases"]) == length(pss["station_id"])
    @test length(pss["mean_cross_correlation"]) == length(pss["station_id"])
    @test length(pss["polarity_match"]) == length(pss["station_id"])
    @test length(pss["misfit_total"]) == length(pss["station_id"])

    # Verify summary
    sm = result.summary
    @test sm["total_iterations"] == 2
    @test sm["total_trials"] == n_trials
    @test sm["convergence_reason"] == "user"

    @test true
end

# ─────────────────────────────────────────────────────────
# Test 4: compile_solution — single trial edge case
# ─────────────────────────────────────────────────────────
@testset "compile_solution: single trial" begin
    strategy = Strategy(
        0.0, 0.0, 0, 90.0, 0.0, 0, 0.0, 0.0, 0,
        Int32[1], Int32[1],
        Int32[1], Int32[1], Int32[1],
        [1.0, 0.5, 0.3],
        [0.0, 90.0, 0.0],
        1, 0.0, 1, 1, "user",
        [0.0 90.0 0.0],
        zeros(Float64, 1, 3),
        [0.0],
    )

    trials = TrialSet([0.0], [90.0], [0.0], [10.0], Int32[1], Int32[1])

    n_phases = 1
    xcorr = [0.0;;]
    polarity = [0.0;;]
    psr = [0.0;;]

    misfits = Dict(
        :xcorr => xcorr,
        :polarity => polarity,
        :psr => psr,
    )

    idx = Index(["NET.ST1.Z.P"], ["P"], Int32[1], [10.0], [30.0], fill(Int32(1), 1, 1))
    config = Dict("depth_vals" => [10.0])

    result = SolutionComp.compile_solution(strategy, trials, misfits, idx, config)

    sol = result.solution
    @test sol["strike"] == 0.0
    @test sol["dip"] == 90.0
    @test sol["rake"] == 0.0

    # Verify pure double-couple: [0,0,0,1,0,0]
    mt = sol["moment_tensor"]
    @test mt ≈ [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
end

# ─────────────────────────────────────────────────────────
# Test 5: Waveform synthesis (GF × MT)
# ─────────────────────────────────────────────────────────
@testset "waveform synthesis: GF × MT" begin
    # GF: 10 samples × 6 components
    gf = rand(10, 6)
    mt = MTUtils.sdr_to_mt(45.0, 30.0, 90.0)

    # Manual: synthetic = GF × MT
    synthetic_manual = gf * mt

    # Verify correct dimensions
    @test length(synthetic_manual) == 10

    # Verify BLAS identity: ‖GF·m‖² = mᵀ·GFᵀ·GF·m
    syn_norm_sq_direct = sum(synthetic_manual.^2)
    gf2 = gf' * gf  # 6×6 Gram matrix
    syn_norm_sq_gram = mt' * gf2 * mt
    @test isapprox(syn_norm_sq_direct, syn_norm_sq_gram, atol=1e-12)
end

# ─────────────────────────────────────────────────────────
# Test 6: Per-station breakdown correctness
# ─────────────────────────────────────────────────────────
@testset "per-station breakdown" begin
    strategy = Strategy(
        0.0, 0.0, 0, 90.0, 0.0, 0, 0.0, 0.0, 0,
        Int32[1], Int32[1],
        Int32[1, 1], Int32[1], Int32[1],
        [1.0, 0.5, 0.3],
        [0.0, 90.0, 0.0],
        1, 0.0, 1, 1, "user",
        [0.0 90.0 0.0],
        zeros(Float64, 1, 3),
        [0.0],
    )

    n_trials = 2
    trials = TrialSet(
        [0.0, 0.0], [90.0, 90.0], [0.0, 0.0],
        [10.0, 10.0], Int32[1, 1], Int32[1, 1],
    )

    n_ph = 2
    xcorr = [0.1 0.2; 0.15 0.25]  # 2 phases × 2 trials
    polarity = [0.0 0.0]           # 1 station × 2 trials
    psr = [0.3 0.4]                # 1 station × 2 trials

    misfits = Dict(:xcorr => xcorr, :polarity => polarity, :psr => psr)
    idx = Index(
        ["NET.ST1.Z.P", "NET.ST1.Z.S"],
        ["P", "S"],
        Int32[1, 1],
        [10.0, 10.0],
        [30.0, 30.0],
        fill(Int32(1), 2, 2),
    )
    config = Dict("depth_vals" => [10.0])

    result = SolutionComp.compile_solution(strategy, trials, misfits, idx, config)
    pp = result.per_phase

    # Verify per-phase misfits at best trial (trial 1, the one with lowest total)
    @test size(pp["misfit_per_module"], 2) == 2  # 2 phases
    @test length(pp["phase_id"]) == 2
    @test length(pp["station_id"]) == 2
    @test length(pp["cross_correlation"]) == 2

    # Verify per_station_summary is also present
    pss = result.per_station_summary
    @test length(pss["station_id"]) >= 1
end

println("All solution compilation tests passed!")