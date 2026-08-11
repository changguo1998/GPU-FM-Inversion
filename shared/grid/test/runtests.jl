# Grid module tests: expand_axis, generate_trials, refine_strategy, prompt_operator.
# Run from the repo root environment: jula --project=. -e 'include("shared/grid/test/runtests.jl")'

using Grid
using IO
using Test

# === Axis expansion ===

@testset "expand_axis" begin
    @test Grid.expand_axis(0.0, 5.0, Int32(3)) == [0.0, 5.0, 10.0]
    # n <= 0 → fixed axis (single value)
    @test Grid.expand_axis(30.0, 5.0, Int32(0)) == [30.0]
    @test Grid.expand_axis(30.0, 5.0, Int32(-2)) == [30.0]
end

@testset "default_grid matches IO.DEFAULT_GRID" begin
    g = Grid.default_grid()
    @test g == IO.DEFAULT_GRID
    @test g.nstrike == 72 && g.ndip == 19 && g.nrake == 37
end

# === generate_trials ===

# Full-space default strategy: 72×19×37 strikes/dips/rakes × 3 depths × 1 freq
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
        Int32(0),
    )
end

@testset "generate_trials count and nesting" begin
    trials = Grid.generate_trials(full_strategy())
    n = 72 * 19 * 37 * 3 * 1
    @test length(trials.strike_idx) == n
    # nesting: strike outermost, freq innermost; all params are 1-based indices
    @test trials.strike_idx[1] == 1 && trials.strike_idx[n] == 72  # 72-axis
    @test trials.freq_idx[1] == 1
    # cycles within each (strike,dip,rake) combo (strike → dip → rake → depth → freq)
    @test trials.depth_idx[1:3] == Int32[1, 2, 3]
    @test trials.depth_idx == Int32.(repeat([1, 2, 3], n ÷ 3))
end

@testset "generate_trials empty indices default to [1]" begin
    s = IO.Strategy(
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
        Int32(0),
    )
    trials = Grid.generate_trials(s)
    # SDR 3×3×3 × depth[1] × freq[1]
    @test length(trials.strike_idx) == 27
    @test all(trials.depth_idx .== 1)
    @test all(trials.freq_idx .== 1)
end

# === refine_strategy ===

@testset "refine_strategy centering + halving + subsets" begin
    current = full_strategy()
    best = Grid.TrialResult(
        [100.0, 30.0, 10.0],   # sdr
        Int32(2),              # best depth_idx
        Int32(1),              # best freq_idx
        0.5,
        [0.4, 0.5, 0.6],       # depth_misfits → within 1.2×0.5 = 0.6 → idx 1,2,3
        [0.5],                 # freq_misfits → idx 1
    )
    next = Grid.refine_strategy(current, best)
    @test next isa IO.Strategy
    # steps halved
    @test next.dstrike == 2.5 && next.ddip == 2.5 && next.drake == 2.5
    # centered: expanded 3-wide axis puts best at index 2
    strikes = Grid.expand_axis(next.strike0, next.dstrike, next.nstrike)
    @test strikes[2] == 100.0
    dips = Grid.expand_axis(next.dip0, next.ddip, next.ndip)
    @test dips[2] == 30.0
    rakes = Grid.expand_axis(next.rake0, next.drake, next.nrake)
    @test rakes[2] == 10.0
    @test next.nstrike == 3 && next.ndip == 3 && next.nrake == 3
    # depth subset within 1.2× threshold
    @test next.depth_indices == Int32[1, 2, 3]
    @test next.freq_indices == Int32[1]
    @test next.iteration == 1
end

@testset "refine_strategy domain clamping + empty-subset fallback" begin
    current = full_strategy()
    # best strike at boundary 0 → strike0 would be -2.5, mod → 357.5
    # best dip at 0 → dip0 clamps to 0.0
    best = Grid.TrialResult(
        [0.0, 0.0, 90.0],
        Int32(1),
        Int32(1),
        1.0,
        [1.0, 10.0, 10.0],   # only idx 1 within 1.2×1.0
        [1.0, 10.0],
    )
    next = Grid.refine_strategy(current, best)
    s = Grid.expand_axis(next.strike0, next.dstrike, next.nstrike)
    @test s[2] == 0.0 || s[2] == 360.0 || isapprox(s[2], 0.0)  # mod-wrapped center
    @test next.strike0 >= 0.0 && next.strike0 < 360.0
    @test next.dip0 == 0.0                 # clamped, not negative
    @test next.rake0 == 87.5               # -90+90 → rake0 = 90-2.5 = 87.5
    # freq empty subset → fallback to best index
    @test next.freq_indices == Int32[1]
end

# === prompt_operator ===

@testset "prompt_operator" begin
    cur = full_strategy()
    # 'y' → continue
    @test Grid.prompt_operator(
        [100.0, 30.0, 10.0],
        0.5,
        cur;
        io_in = IOBuffer("y\n"),
        io_out = IOBuffer(),
    )
    # anything else → stop
    @test !Grid.prompt_operator(
        [100.0, 30.0, 10.0],
        0.5,
        cur;
        io_in = IOBuffer("n\n"),
        io_out = IOBuffer(),
    )
    @test !Grid.prompt_operator(
        [100.0, 30.0, 10.0],
        0.5,
        cur;
        io_in = IOBuffer("\n"),
        io_out = IOBuffer(),
    )
end
