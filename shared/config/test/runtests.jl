using Config
using Misfit
using Test

@testset "duration interface" begin
    @test isdefined(Config, :durations)
    @test_throws Config.ConfigError Config.durations()
end

@testset "objective expression registration" begin
    p_observed = Misfit.observed(Misfit.P; band = (0.5, 2.0), window = (-2, 8))
    p_synthetic = Misfit.synthetic(Misfit.P; band = (0.5, 2.0), window = (-2, 8))
    s_observed = Misfit.observed(Misfit.S; band = (0.5, 2.0), window = (-2, 8))
    s_synthetic = Misfit.synthetic(Misfit.S; band = (0.5, 2.0), window = (-2, 8))
    expr = 1 - Misfit.maxCC(p_observed, p_synthetic; maxlag = 3)

    Config.@objective T_Objective = expr
    Config.@objective T_XcorrS = 1 - Misfit.maxCC(s_observed, s_synthetic; maxlag = 3)
    Config.@objective T_LagP = Misfit.lagCC(p_observed, p_synthetic; maxlag = 3)
    Config.@objective T_Psr = abs2(
        log(Misfit.rms(s_observed) / Misfit.rms(p_observed)) -
        log(Misfit.rms(s_synthetic) / Misfit.rms(p_synthetic)),
    )
    observed_polarity = Misfit.ampScale(p_observed) * Misfit.signScale(p_observed)
    synthetic_polarity = Misfit.ampScale(p_synthetic) * Misfit.signScale(p_synthetic)
    Config.@objective T_Polarity = abs(
        observed_polarity / Misfit.energy(observed_polarity)^0.5 -
        synthetic_polarity / Misfit.energy(synthetic_polarity)^0.5,
    )
    @test Config.objective(:T_Objective) === expr
    @test Config.objectives()[:T_Objective] === expr
    @test_throws ErrorException Config.objective!(:T_Objective, expr)
    @test_throws MethodError Config.objective!(:T_Invalid, 1.0)
end

@testset "objective compilation" begin
    Config.freq_bands() = [(0.5, 2.0)]
    Config.compile_objectives!()
    @test Config.phase_type(:T_Objective) == "P"
    @test Config.output_field(:T_Objective) == Misfit.Xcorr.CC_MAX
    @test Config.operator_module(:T_Objective) === Misfit.Xcorr
    instance = Config._instance_module(:T_Objective)
    @test instance.trim() == [-2.0, 8.0]
    @test instance.max_lag_periods() == 3.0
    @test instance.filter_order() == 4
    @test instance.band_low() == Int32[1]
    @test instance.band_high() == Int32[2]
    @test "T_Objective" in Config.misfit_modules()
    @test Config.output_field(:T_LagP) == Misfit.Xcorr.BEST_LAG
    @test Config.is_composed(:T_Psr)
    @test Config.bases_of(:T_Psr) == [:T_Objective, :T_XcorrS]
    @test Config.operator_module(:T_Psr) === Misfit.Psr
    @test Config.is_composed(:T_Polarity)
    @test Config.bases_of(:T_Polarity) == [:T_Objective]
    @test Config.output_field(:T_Polarity) == Misfit.Polarity.NORMALIZED_L1
    @test isnothing(Config.compile_objectives!())

    unsupported = Misfit.energy(Misfit.input(:waveform))
    @test_throws ArgumentError Config._compile_objective!(:T_Unsupported, unsupported)
end

@testset "use_misfit! base registration" begin
    Config.use_misfit!(
        :T2_XcorrP,
        operator = Misfit.Xcorr,
        phase = "P",
        output = Misfit.Xcorr.CC_MAX,
    )
    @test Config.phase_type(:T2_XcorrP) == "P"
    @test Config.output_field(:T2_XcorrP) == :cc_max
    @test Config.operator_module(:T2_XcorrP) === Misfit.Xcorr
    @test !Config.is_composed(:T2_XcorrP)
    @test Config.channel_of(:T2_XcorrP) === nothing
    @test "T2_XcorrP" in Config.misfit_modules()
    # instance module allows per-instance override
    Config.T2_XcorrP.trim() = [-2.0, 5.0]
    @test Config.T2_XcorrP.trim() == [-2.0, 5.0]
end

@testset "use_misfit! validation" begin
    @test_throws Exception Config.use_misfit!(
        :T2_Bad,
        operator = Misfit.Xcorr,
        phase = "P",
        output = :nonexistent,
    )
end

@testset "use_misfit! composed registration" begin
    Config.use_misfit!(
        :T2_Rel,
        operator = Misfit.Xcorr,  # placeholder; T4 swaps in StdDev
        bases = [:T2_XcorrP],
        output = Misfit.Xcorr.CC_MAX,
    )
    @test Config.is_composed(:T2_Rel)
    @test Config.bases_of(:T2_Rel) == [:T2_XcorrP]
    @test !isdefined(Config, :T2_Rel)  # Level 2 creates no instance module
end

@testset "channel filter" begin
    Config.use_misfit!(
        :T2_XcorrSH,
        operator = Misfit.Xcorr,
        phase = "S",
        channel = "H",
        output = Misfit.Xcorr.BEST_LAG,
    )
    @test Config.channel_of(:T2_XcorrSH) == "H"
end
