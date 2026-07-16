using Config
using Misfit
using Test

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
