using Misfit
using Test

@testset "Xcorr outputs" begin
    @test :cc_max in Misfit.Xcorr.outputs()
    @test :best_lag in Misfit.Xcorr.outputs()
    @test Misfit.Xcorr.CC_MAX == :cc_max
    @test Misfit.Xcorr.BEST_LAG == :best_lag
    @test Misfit.Xcorr.is_freq_dependent() == true
end

@testset "Xcorr params" begin
    @test !isdefined(Misfit.Xcorr, :select_threshold)
    @test !isdefined(Misfit.Xcorr, :deselect_threshold)
    @test !isdefined(Misfit.Xcorr, :maxlag_factor)
    @test isdefined(Misfit.Xcorr, :max_lag_periods)
end

@testset "Xcorr per-lag reductions" begin
    dt = 0.1
    arrival = 50
    pre_p = 2.0
    post_p = 5.0
    band_high = 0.5
    max_lag_p = 3.0
    nt_win = round(Int, pre_p / band_high / dt) + round(Int, post_p / band_high / dt) + 1
    obs_win = randn(nt_win)
    gf_full = randn(200, 6)
    obs_n2, synamp_lag, dog_lag =
        Misfit.Xcorr.preprocess(gf_full, obs_win, dt, arrival, pre_p, post_p, band_high, max_lag_p)
    L = size(synamp_lag, 3)
    @test L == 2 * round(Int, max_lag_p / band_high / dt) + 1
    @test size(synamp_lag) == (6, 6, L)
    @test size(dog_lag) == (6, L)
    @test obs_n2 ≈ sum(abs2, obs_win)
    center = (L + 1) ÷ 2
    @test synamp_lag[:, :, center] ≈ synamp_lag[:, :, center]'   # lag=0 symmetric
end

@testset "Polarity outputs" begin
    @test :syn_sign in Misfit.Polarity.outputs()
    @test :dot_value in Misfit.Polarity.outputs()
    @test Misfit.Polarity.SYN_SIGN == :syn_sign
    @test Misfit.Polarity.DOT_VALUE == :dot_value
    @test Misfit.Polarity.is_freq_dependent() == false
end

@testset "Polarity params" begin
    @test !isdefined(Misfit.Polarity, :trim)
    @test isdefined(Misfit.Polarity, :source_duration)
end
