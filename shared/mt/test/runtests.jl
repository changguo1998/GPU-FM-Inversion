using Test
using MT
using Random

println("Running MT tests...")

# ─── Test 1: Pure double-couple (strike=0, dip=90, rake=0) ───
@testset "Pure double-couple" begin
    mt = sdr_to_mt(0.0, 90.0, 0.0)
    expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test maximum(abs.(mt .- expected)) < 1e-12
    @test length(mt) == 6
end

# ─── Test 2: Batch vs single on 100 random SDRs ───
@testset "Batch matches single (100 random SDRs)" begin
    rng = MersenneTwister(42)
    N = 100
    strikes = [rand(rng) * 360.0 for _ in 1:N]
    dips = [rand(rng) * 90.0 for _ in 1:N]
    rakes = .-90.0 .+ rand(rng, N) .* 180.0

    batch = sdr_to_mt_batch(strikes, dips, rakes)
    @test size(batch) == (6, N)

    for i in 1:N
        single = sdr_to_mt(strikes[i], dips[i], rakes[i])
        @test maximum(abs.(batch[:, i] .- single)) < 1e-15
    end
end

# ─── Test 3: Boundary values ───
@testset "Boundary values" begin
    # strike=0 vs strike=360 should give same result
    mt0 = sdr_to_mt(0.0, 45.0, 30.0)
    mt360 = sdr_to_mt(360.0, 45.0, 30.0)
    @test maximum(abs.(mt0 .- mt360)) < 1e-12

    # dip=0
    mt = sdr_to_mt(45.0, 0.0, 30.0)
    @test length(mt) == 6

    # dip=90
    mt = sdr_to_mt(45.0, 90.0, 30.0)
    @test length(mt) == 6

    # rake=-90
    mt = sdr_to_mt(45.0, 45.0, -90.0)
    @test length(mt) == 6

    # rake=90
    mt = sdr_to_mt(45.0, 45.0, 90.0)
    @test length(mt) == 6

    # All combinations at extremes
    for s in (0.0, 90.0, 180.0, 270.0, 360.0)
        for d in (0.0, 45.0, 90.0)
            for r in (-90.0, 0.0, 90.0)
                mt = sdr_to_mt(s, d, r)
                @test length(mt) == 6
                @test all(isfinite, mt)
            end
        end
    end
end

println("All tests passed!")
