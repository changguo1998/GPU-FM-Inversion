using Misfit
using Test

@testset "Xcorr outputs" begin
    @test :cc_max in Misfit.Xcorr.outputs()
    @test :best_lag in Misfit.Xcorr.outputs()
    @test Misfit.Xcorr.CC_MAX == :cc_max
    @test Misfit.Xcorr.BEST_LAG == :best_lag
    @test Misfit.Xcorr.is_freq_dependent() == true
end

@testset "Polarity outputs" begin
    @test :syn_sign in Misfit.Polarity.outputs()
    @test :dot_value in Misfit.Polarity.outputs()
    @test Misfit.Polarity.SYN_SIGN == :syn_sign
    @test Misfit.Polarity.DOT_VALUE == :dot_value
    @test Misfit.Polarity.is_freq_dependent() == false
end
