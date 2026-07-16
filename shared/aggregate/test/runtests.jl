using Aggregate
using Statistics
using Test

@testset "StdDev outputs" begin
    @test :relative_offset in Aggregate.StdDev.outputs()
    @test :mean in Aggregate.StdDev.outputs()
end

@testset "XCorr extractors" begin
    inter = Dict("cc_max" => [0.8 0.6; 0.9 0.5], "best_lag" => Int32[2 -1; 0 3])
    ctx = (dt = 0.5,)
    cc = Aggregate.EXTRACTORS[(:Xcorr, :cc_max)](inter, ctx)
    @test cc ≈ [0.2 0.4; 0.1 0.5]
    sh = Aggregate.EXTRACTORS[(:Xcorr, :best_lag)](inter, ctx)
    @test sh ≈ [1.0 -0.5; 0.0 1.5]
end

@testset "Polarity extractors" begin
    inter = Dict("syn_sign" => Int8[1 -1; 1 1], "dot_value" => [0.5 -0.3; 1.2 0.0])
    ctx = (obs_pol = Int8[1 -1; 1 -1],)
    m = Aggregate.EXTRACTORS[(:Polarity, :syn_sign)](inter, ctx)
    @test m ≈ [0.0 0.0; 0.0 1.0]  # match=0, mismatch=1
end

@testset "StdDev composer" begin
    # two bases, each 1 phase × 2 trials, both belong to station 1
    base_misfits = [[1.0 2.0], [3.0 4.0]]
    base_station_idx = [Int32[1], Int32[1]]
    ctx = (N_stations = 1,)
    out = Aggregate.COMPOSERS[:StdDev](base_misfits, base_station_idx, ctx)
    @test :relative_offset in keys(out)
    @test out[:relative_offset][1, 1] ≈ std([1.0, 3.0])
    @test out[:relative_offset][1, 2] ≈ std([2.0, 4.0])
end
