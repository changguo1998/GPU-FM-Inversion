# Search module tests: expand_axis and generate_trials.
# Run from the repo root environment: julia --project=. -e 'include("shared/search/test/runtests.jl")'

using IO
using Search
using Test

@testset "expand_axis" begin
    @test Search.expand_axis(0.0, 5.0, Int32(3)) == [0.0, 5.0, 10.0]
    @test Search.expand_axis(30.0, 5.0, Int32(0)) == [30.0]
    @test Search.expand_axis(30.0, 5.0, Int32(-2)) == [30.0]
end

@testset "default_grid matches IO.DEFAULT_GRID" begin
    grid = Search.default_grid()
    @test grid == IO.DEFAULT_GRID
    @test grid.nstrike == 72 && grid.ndip == 19 && grid.nrake == 37
end

function full_strategy()
    return IO.Strategy(
        IO.DEFAULT_GRID.strike0,
        IO.DEFAULT_GRID.dstrike,
        IO.DEFAULT_GRID.nstrike,
        IO.DEFAULT_GRID.dip0,
        IO.DEFAULT_GRID.ddip,
        IO.DEFAULT_GRID.ndip,
        IO.DEFAULT_GRID.rake0,
        IO.DEFAULT_GRID.drake,
        IO.DEFAULT_GRID.nrake,
        Int32[1, 2, 3],
        Int32[1],
        Int32[1, 2, 3],
        Int32(0),
    )
end

@testset "generate_trials count and nesting" begin
    trials = Search.generate_trials(full_strategy())
    n_trials = 72 * 19 * 37 * 3 * 1 * 3
    @test length(trials.strike_idx) == n_trials
    @test trials.strike_idx[1] == 1 && trials.strike_idx[n_trials] == 72
    @test trials.freq_idx[1] == 1
    @test trials.depth_idx[1:9] == Int32[1, 1, 1, 2, 2, 2, 3, 3, 3]
    @test trials.duration_idx[1:9] == Int32[1, 2, 3, 1, 2, 3, 1, 2, 3]
end

@testset "generate_trials empty indices default to [1]" begin
    strategy = IO.Strategy(
        0.0,
        5.0,
        Int32(3),
        0.0,
        5.0,
        Int32(3),
        0.0,
        5.0,
        Int32(3),
        Int32[],
        Int32[],
        Int32[],
        Int32(0),
    )
    trials = Search.generate_trials(strategy)
    @test length(trials.strike_idx) == 27
    @test all(trials.depth_idx .== 1)
    @test all(trials.freq_idx .== 1)
    @test all(trials.duration_idx .== 1)
end

@testset "budgeted plan stays within budget and preserves global indices" begin
    strategy = IO.Strategy(
        0.0,
        5.0,
        Int32(8),
        0.0,
        5.0,
        Int32(5),
        -90.0,
        5.0,
        Int32(6),
        Int32[2, 4, 6],
        Int32[1],
        Int32[3, 5, 7],
        Int32(0),
    )
    plan = Search.budgeted_plan(strategy, 96)
    trials = Search.generate_trials(plan)
    @test length(trials.strike_idx) <= 96
    @test plan.strike_indices[1] == 1
    @test all(x -> 1 <= x <= 8, plan.strike_indices)
    @test all(x -> x in strategy.depth_indices, plan.depth_indices)
    @test all(x -> x in strategy.duration_indices, plan.duration_indices)
    @test extrema(plan.depth_indices) == (Int32(2), Int32(6))
    @test extrema(plan.duration_indices) == (Int32(3), Int32(7))
end

@testset "budgeted plan reaches the complete space" begin
    strategy = full_strategy()
    full_count = 72 * 19 * 37 * 3 * 1 * 3
    plan = Search.budgeted_plan(strategy, full_count)
    trials = Search.generate_trials(plan)
    @test length(trials.strike_idx) == full_count
    @test plan.strike_indices == Int32.(1:72)
    @test plan.dip_indices == Int32.(1:19)
    @test plan.rake_indices == Int32.(1:37)
end

@testset "budgeted plan validates minimum budget" begin
    @test_throws ArgumentError Search.budgeted_plan(full_strategy(), 0)
    @test_throws ArgumentError Search.budgeted_plan(full_strategy(), 31)
    plan = Search.budgeted_plan(full_strategy(), 32)
    @test length(Search.generate_trials(plan).strike_idx) == 32
    @test plan.strike_indices == Int32[1, 37]
    @test plan.dip_indices == Int32[1, 19]
    @test plan.rake_indices == Int32[1, 37]
end
