using Test
include("../src/trial_gen.jl")
using .TrialGen

# ─────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────

depth_vals = Float64[5.0, 10.0, 15.0, 20.0, 25.0]

# ─────────────────────────────────────────────────────────
# Test 1: All axes varying (3×3×3 SDR, 2 depths, 1 freq)
# ─────────────────────────────────────────────────────────
@testset "All axes varying — 3×3×3 SDR × 2 depths × 1 freq" begin
    strat = GridStrategy(
        45.0, 10.0, 3,
        30.0, 8.0, 3,
        90.0, 8.0, 3,
        Int32[2, 4],    # depth indices
        Int32[1],        # freq index
        1,
    )
    trials = generate_trials(strat, depth_vals)

    # Count: 3 × 3 × 3 × 2 × 1 = 54
    @test length(trials.strike) == 54
    @test length(trials.dip) == 54
    @test length(trials.rake) == 54
    @test length(trials.depth) == 54
    @test length(trials.depth_idx) == 54
    @test length(trials.freq_idx) == 54

    # Check first trial (outermost strike=45, dip=30, rake=90, depth_idx=2, freq=1)
    @test trials.strike[1]   ≈ 45.0
    @test trials.dip[1]      ≈ 30.0
    @test trials.rake[1]     ≈ 90.0
    @test trials.depth_idx[1] == 2
    @test trials.depth[1]    ≈ 10.0   # depth_vals[2]
    @test trials.freq_idx[1]  == 1

    # Order (innermost→outermost): freq→depth→rake→dip→strike
    # Strikes: [45, 55, 65] — varies slowest
    @test trials.strike[1:3]   ≈ [45.0, 45.0, 45.0]

    # Depth varies fastest (only 1 freq): trial 1→2 changes depth
    @test trials.depth_idx[1] == 2
    @test trials.depth_idx[2] == 4
    @test trials.depth_idx[3] == 2  # depth wraps, rake changes

    # Rake changes after 2 depths: trials 1-2 rake=90, trial 3 rake=98
    @test trials.rake[1]   ≈ 90.0
    @test trials.rake[3]   ≈ 98.0  # after 2 depths, rake increments

    # Dip changes after 3 rakes × 2 depths = 6 trials
    @test trials.dip[1]    ≈ 30.0
    @test trials.dip[7]    ≈ 38.0   # trial 7: first dip change

    # Strike changes after 3 dips × 3 rakes × 2 depths = 18 trials
    @test trials.strike[1]  ≈ 45.0
    @test trials.strike[19] ≈ 55.0  # trial 19: second strike value

    # All freq_idx should be 1 (only one freq band)
    @test all(f -> f == 1, trials.freq_idx)
end

# ─────────────────────────────────────────────────────────
# Test 2: Fixed axes — single trial
# ─────────────────────────────────────────────────────────
@testset "Fixed axes — single trial" begin
    strat = GridStrategy(
        45.0, 10.0, 0,    # fixed strike
        30.0, 8.0, 0,     # fixed dip
        90.0, 8.0, 0,     # fixed rake
        Int32[],           # empty → best_depth_index
        Int32[],           # empty → default freq
        3,                 # best_depth_index
    )
    trials = generate_trials(strat, depth_vals)

    @test length(trials.strike) == 1
    @test trials.strike[1]   ≈ 45.0
    @test trials.dip[1]      ≈ 30.0
    @test trials.rake[1]     ≈ 90.0
    @test trials.depth_idx[1] == 3
    @test trials.depth[1]    ≈ 15.0   # depth_vals[3]
    @test trials.freq_idx[1]  == 1
end

# ─────────────────────────────────────────────────────────
# Test 3: Deterministic order
# ─────────────────────────────────────────────────────────
@testset "Deterministic order — repeated calls identical" begin
    strat = GridStrategy(
        0.0, 10.0, 5,
        30.0, 5.0, 3,
        -90.0, 5.0, 3,
        Int32[1, 2, 3],
        Int32[1, 2],
        1,
    )

    t1 = generate_trials(strat, depth_vals)
    t2 = generate_trials(strat, depth_vals)

    @test t1.strike   == t2.strike
    @test t1.dip      == t2.dip
    @test t1.rake     == t2.rake
    @test t1.depth    == t2.depth
    @test t1.depth_idx == t2.depth_idx
    @test t1.freq_idx  == t2.freq_idx
end

# ─────────────────────────────────────────────────────────
# Test 4: Only strike varies
# ─────────────────────────────────────────────────────────
@testset "Only strike varies" begin
    strat = GridStrategy(
        0.0, 10.0, 5,    # 5 strike values
        30.0, 5.0, 0,    # fixed dip
        -90.0, 5.0, 0,   # fixed rake
        Int32[2],         # fixed depth
        Int32[1],          # fixed freq
        1,
    )
    trials = generate_trials(strat, depth_vals)

    @test length(trials.strike) == 5
    @test trials.strike ≈ [0.0, 10.0, 20.0, 30.0, 40.0]
    @test all(d -> d ≈ 30.0, trials.dip)
    @test all(r -> r ≈ -90.0, trials.rake)
    @test all(d -> d ≈ 10.0, trials.depth)
    @test all(d -> d == 2, trials.depth_idx)
end

# ─────────────────────────────────────────────────────────
# Test 5: Multiple frequencies
# ─────────────────────────────────────────────────────────
@testset "Multiple freq bands" begin
    strat = GridStrategy(
        0.0, 10.0, 2,    # 2 strikes
        30.0, 5.0, 0,
        -90.0, 5.0, 0,
        Int32[1],         # 1 depth
        Int32[1, 2, 3],   # 3 freqs
        1,
    )
    trials = generate_trials(strat, depth_vals)

    @test length(trials.strike) == 6  # 2 × 1 × 1 × 1 × 3
    # Order: strike→dip→rake→depth→freq (freq innermost)
    @test trials.freq_idx[1:3] == [1, 2, 3]   # same strike
    @test trials.freq_idx[4:6] == [1, 2, 3]   # next strike
    @test trials.strike[1:3] ≈ [0.0, 0.0, 0.0]
    @test trials.strike[4:6] ≈ [10.0, 10.0, 10.0]
end

# ─────────────────────────────────────────────────────────
# Test 6: Edge case — zero trials? (should still work)
# ─────────────────────────────────────────────────────────
@testset "Edge — empty depth_indices with best_depth_index" begin
    strat = GridStrategy(
        0.0, 10.0, 2,
        30.0, 5.0, 0,
        -90.0, 5.0, 0,
        Int32[],    # empty → fall back to best_depth_index
        Int32[],    # empty → fall back to default freq
        5,          # best_depth_index = 5
    )
    trials = generate_trials(strat, depth_vals)

    @test length(trials.strike) == 2
    @test trials.depth_idx == Int32[5, 5]
    @test trials.depth ≈ [25.0, 25.0]
    @test trials.freq_idx == Int32[1, 1]
end

# ─────────────────────────────────────────────────────────
# Test 7: Large grid stress test
# ─────────────────────────────────────────────────────────
@testset "Large grid — trial count matches product" begin
    n_strike = 5
    n_dip = 4
    n_rake = 6
    n_depth = 3
    n_freq = 2

    strat = GridStrategy(
        0.0, 10.0, n_strike,
        30.0, 5.0, n_dip,
        -90.0, 5.0, n_rake,
        Int32[1, 2, 3],
        Int32[1, 2],
        1,
    )
    trials = generate_trials(strat, depth_vals)

    expected = n_strike * n_dip * n_rake * n_depth * n_freq  # 5×4×6×3×2 = 720
    @test length(trials.strike) == expected

    # Verify values at specific positions — order: freq→depth→rake→dip→strike
    # Trial 1: (s=0, d=30, r=-90, didx=1, fidx=1)
    @test trials.strike[1] ≈ 0.0
    @test trials.dip[1]    ≈ 30.0
    @test trials.rake[1]   ≈ -90.0
    @test trials.depth_idx[1] == 1
    @test trials.freq_idx[1] == 1

    # After 2 freqs: depth increments (innermost→outermost)
    @test trials.freq_idx[2] == 2
    @test trials.depth_idx[3] == 2   # freq wrapped, depth advances
    @test trials.freq_idx[3] == 1    # freq restarts

    # After 2 freqs × 3 depths = 6 trials: rake advances
    @test trials.rake[7] ≈ -85.0
    @test trials.strike[7] ≈ 0.0   # strike still first
end

# ─────────────────────────────────────────────────────────
# Test 8: Out-of-bounds depth index
# ─────────────────────────────────────────────────────────
@testset "Edge — out-of-bounds depth index returns NaN" begin
    strat = GridStrategy(
        0.0, 10.0, 1,
        30.0, 5.0, 0,
        -90.0, 5.0, 0,
        Int32[99],   # out of bounds
        Int32[1],
        1,
    )
    trials = generate_trials(strat, depth_vals)

    @test length(trials.strike) == 1
    @test isnan(trials.depth[1])
    @test trials.depth_idx[1] == 99
end

println("All trial generation tests passed!")