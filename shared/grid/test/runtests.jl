# Grid module tests: expand_axis and generate_trials.
# Run from the repo root environment: julia --project=. -e 'include("shared/grid/test/runtests.jl")'

using Grid
using IO
using Test

@testset "expand_axis" begin
    @test Grid.expand_axis(0.0, 5.0, Int32(3)) == [0.0, 5.0, 10.0]
    @test Grid.expand_axis(30.0, 5.0, Int32(0)) == [30.0]
    @test Grid.expand_axis(30.0, 5.0, Int32(-2)) == [30.0]
end

@testset "default_grid matches IO.DEFAULT_GRID" begin
    grid = Grid.default_grid()
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
    trials = Grid.generate_trials(full_strategy())
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
    trials = Grid.generate_trials(strategy)
    @test length(trials.strike_idx) == 27
    @test all(trials.depth_idx .== 1)
    @test all(trials.freq_idx .== 1)
    @test all(trials.duration_idx .== 1)
end
