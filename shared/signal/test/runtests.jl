using Signal
using Test

@testset "demean/detrend/taper" begin
    wf = [1.0, 2.0, 3.0, 4.0, 5.0]
    Signal.demean!(wf)
    @test abs(sum(wf)) < 1e-9                        # mean ≈ 0 after demean
    wf2 = [1.0, 3.0, 5.0, 7.0, 9.0]                  # linear trend
    Signal.detrend!(wf2)
    @test maximum(abs.(wf2)) < 1e-9                  # detrended ≈ 0
    wf3 = ones(100)
    Signal.taper!(wf3; frac = 0.1)
    @test wf3[1] < 1.0 && wf3[50] ≈ 1.0 && wf3[end] < 1.0
end

@testset "preprocess_waveform!" begin
    dt = 0.1
    wf = randn(500)
    low_cut, high_cut = 0.1, 0.5
    out = Signal.preprocess_waveform!(copy(wf), dt, low_cut, high_cut)
    @test length(out) == length(wf)                  # full waveform, no trim
    @test all(!isnan, out)
    # do_bandpass=false: clean only, no filter
    out2 = Signal.preprocess_waveform!(copy(wf), dt, low_cut, high_cut; do_bandpass = false)
    @test length(out2) == length(wf)
end

@testset "trim_time_window! non-symmetric" begin
    dt = 0.1
    obs = collect(1.0:200.0)                          # long enough, no clamp
    gf = reshape(collect(1.0:1200.0), 200, 6)
    arrival = 50
    # pre_periods=2, post_periods=5, band_high=0.5 -> pre_sec=4 (40 samples), post_sec=10 (100)
    ot, gt = Signal.trim_time_window!(obs, gf, dt, arrival, 2.0, 5.0, 0.5)
    pre_n = round(Int, 2.0 / 0.5 / dt)               # 40
    post_n = round(Int, 5.0 / 0.5 / dt)              # 100
    @test size(ot, 1) == pre_n + post_n + 1
    @test ot[1] ≈ obs[arrival - pre_n]
    @test ot[end] ≈ obs[arrival + post_n]
    @test size(gt, 2) == 6
end
