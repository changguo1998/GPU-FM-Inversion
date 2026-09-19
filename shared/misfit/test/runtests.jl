using Misfit
using Test

@testset "Target-function operators" begin
    observed = [0.0, 1.0, 0.0, 0.0]
    synthetic = [0.0, 0.0, 1.0, 0.0]
    @test Misfit.maxCC(observed, synthetic; maxlag = 2) ≈ 1.0
    @test Misfit.lagCC(observed, synthetic; maxlag = 2) == -1
    @test Misfit.maxCC(observed, -observed; maxlag = 0) ≈ -1.0
    @test Misfit.maxCC(zeros(4), synthetic; maxlag = 2) == 0.0
    @test Misfit.lagCC(zeros(4), synthetic; maxlag = 2) == 0
    @test_throws ArgumentError Misfit.maxCC(observed, synthetic[1:3])
    @test_throws ArgumentError Misfit.lagCC(observed, synthetic; maxlag = -1)

    x = [-4.0, 0.0, 3.0]
    @test Misfit.energy(x) == 25.0
    @test Misfit.rms(x) ≈ sqrt(25 / 3)
    @test Misfit.ampScale(x) == 3.5
    @test Misfit.signScale(x) == -1.0
    @test Misfit.signScale([3.0, 0.0, -4.0]) == 1.0
    @test Misfit.signScale(fill(-2.0, 3)) == -1.0
    @test_throws ArgumentError Misfit.ampScale(Float64[])
    @test_throws ArgumentError Misfit.signScale(Float64[])
    @test_throws ArgumentError Misfit.rms(Float64[])
end

@testset "Target-function expression nodes" begin
    observed = Misfit.input(:observed)
    synthetic = Misfit.input(:synthetic)

    max_cc = Misfit.maxCC(observed, synthetic; maxlag = 3)
    @test max_cc isa Misfit.CallNode
    @test max_cc.op isa Misfit.MaxCCOp
    @test max_cc.args === (observed, synthetic)
    @test max_cc.kwargs == (maxlag = 3,)

    lag = Misfit.lagCC(observed, synthetic; maxlag = 3)
    @test lag.op isa Misfit.LagCCOp
    @test_throws ArgumentError Misfit.maxCC(observed, synthetic; maxlag = -1)

    @test Misfit.energy(synthetic).op isa Misfit.EnergyOp
    @test Misfit.ampScale(synthetic).op isa Misfit.AmpScaleOp
    @test Misfit.signScale(synthetic).op isa Misfit.SignScaleOp
    @test Misfit.rms(synthetic).op isa Misfit.RMSOp

    cc_misfit = 1 - max_cc
    @test cc_misfit.op isa Misfit.SubtractOp
    @test cc_misfit.args[1] isa Misfit.LiteralNode
    @test cc_misfit.args[1].value == 1
    @test cc_misfit.args[2] === max_cc
    @test (max_cc + 1).op isa Misfit.AddOp
    @test (2 * lag).op isa Misfit.MultiplyOp
    @test (lag / 2).op isa Misfit.DivideOp
    @test (max_cc^2).op isa Misfit.PowerOp
    @test (-lag).op isa Misfit.NegateOp
    @test abs(lag).op isa Misfit.AbsOp
    @test log(Misfit.rms(synthetic)).op isa Misfit.LogOp
    @test sign(lag).op isa Misfit.SignOp

    ratio = Misfit.input(:observed_ratio)
    psr = abs2(log10(Misfit.rms(observed) / Misfit.rms(synthetic)) - ratio)
    @test psr.op isa Misfit.Abs2Op
    @test psr.args[1].op isa Misfit.SubtractOp

    values =
        Dict(:observed => [0.0, 1.0, 0.0], :synthetic => [0.0, 1.0, 0.0], :observed_ratio => 0.0)
    @test Misfit.evaluate(max_cc, values) ≈ 1.0
    @test Misfit.evaluate(cc_misfit, values) ≈ 0.0
    @test Misfit.evaluate(psr, values) ≈ 0.0
    @test_throws KeyError Misfit.evaluate(Misfit.input(:missing), values)

    expressions = AbstractExpr[
        max_cc,
        lag,
        Misfit.energy(synthetic),
        Misfit.ampScale(synthetic),
        Misfit.signScale(synthetic),
        Misfit.rms(synthetic),
        max_cc + 1,
        max_cc - 1,
        max_cc * 2,
        max_cc / 2,
        max_cc ^ 2,
        -max_cc,
        abs(max_cc),
        abs2(max_cc),
        log(Misfit.rms(synthetic)),
        log10(Misfit.rms(synthetic)),
        sign(lag),
    ]
    for expr in expressions
        encoded = Misfit.encode_expression(expr)
        decoded = Misfit.decode_expression(encoded)
        @test Misfit.evaluate(decoded, values) ≈ Misfit.evaluate(expr, values)
    end
    @test_throws ArgumentError Misfit.decode_expression(Dict("kind" => "unknown"))
end

@testset "Waveform source nodes" begin
    observed = Misfit.observed(
        Misfit.P;
        band = (0.5, 2.0),
        window = (-2, 8),
        channel = "Z",
        filter_order = 4,
    )
    synthetic = Misfit.synthetic(
        Misfit.P;
        band = (0.5, 2.0),
        window = (-2, 8),
        channel = "Z",
        filter_order = 4,
    )
    @test observed isa Misfit.WaveformNode
    @test observed.role == :observed
    @test observed.phase == :P
    @test observed.band == (0.5, 2.0)
    @test observed.window == (-2.0, 8.0)
    @test observed.channel == "Z"
    @test observed.filter_order == 4

    decoded = Misfit.decode_expression(Misfit.encode_expression(observed))
    @test decoded == observed
    values = Dict(Misfit._waveform_key(decoded) => [1.0, 2.0])
    @test Misfit.evaluate(decoded, values) == [1.0, 2.0]

    expr = 1 - Misfit.maxCC(observed, synthetic; maxlag = 3.0)
    decoded_expr = Misfit.decode_expression(Misfit.encode_expression(expr))
    inputs = Dict(
        Misfit._waveform_key(observed) => [0.0, 1.0, 0.0],
        Misfit._waveform_key(synthetic) => [0.0, 1.0, 0.0],
    )
    @test Misfit.evaluate(decoded_expr, inputs) ≈ 0.0

    @test_throws ArgumentError Misfit.observed(Misfit.P; band = (2.0, 0.5), window = (-2, 8))
    @test_throws ArgumentError Misfit.observed(Misfit.P; band = (0.5,), window = (-2, 8))
    @test_throws ArgumentError Misfit.observed(Misfit.P; band = (0.5, 2.0), window = (2, -8))
    @test_throws ArgumentError Misfit.observed(
        Misfit.P;
        band = (0.5, 2.0),
        window = (-2, 8),
        filter_order = 0,
    )
end

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

@testset "PSR outputs" begin
    @test :psr_value in Misfit.Psr.outputs()
    @test Misfit.Psr.PSR_VALUE == :psr_value
    @test Misfit.Psr.is_freq_dependent() == true
end

@testset "PSR params" begin
    @test isdefined(Misfit.Psr, :pre_P)
    @test isdefined(Misfit.Psr, :post_P)
    @test isdefined(Misfit.Psr, :pre_S)
    @test isdefined(Misfit.Psr, :post_S)
end
