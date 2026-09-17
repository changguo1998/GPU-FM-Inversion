# report_test.jl — Markdown report generation test.
#
# Uses a compact result.toml fixture to verify report rendering independently
# of the inversion stages.

using Test
using TOML

include("test_util.jl")

@testset "report stage" begin
    mktempdir() do dir
        result = Dict{String, Any}(
            "format_version" => 1,
            "solution" => Dict{String, Any}(
                "strike" => 30.0,
                "dip" => 60.0,
                "rake" => 90.0,
                "depth" => 10.0,
                "duration" => 0.2,
                "freq_idx" => 1.0,
                "duration_idx" => 2.0,
                "misfit" => 0.001,
                "moment_tensor" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            "uncertainty" => Dict{String, Any}(
                "strike_std" => 1.0,
                "dip_std" => 2.0,
                "rake_std" => 3.0,
                "depth_range" => [9.0, 11.0],
                "freq_test_misfit_curve" => [[NaN]],
            ),
            "summary" => Dict{String, Any}(
                "total_iterations" => 1,
                "total_trials" => 12,
                "convergence_reason" => "single iteration",
            ),
            "per_phase" => Dict{String, Any}(
                "phase_id" => ["NET.STA.Z.P", "NET.STA.Z.S"],
                "station_id" => ["NET.STA", "NET.STA"],
                "phase_type" => ["P", "S"],
                "cross_correlation" => [0.99, 0.95],
                "misfit_modules" => ["XcorrP", "XcorrS"],
                "misfit_per_module" => [[0.01, 0.05], [0.02, 0.06]],
            ),
            "per_station_summary" => Dict{String, Any}(
                "station_id" => ["NET.STA"],
                "n_phases" => [2],
                "mean_cross_correlation" => [0.97],
                "misfit_total" => [0.03],
            ),
        )
        open(joinpath(dir, "result.toml"), "w") do io
            TOML.print(io, result)
        end

        run = run_stage_script(
            joinpath("scripts", "report.jl"),
            String[];
            env = Dict("DATA_DIR" => dir),
        )
        @test run.ok
        report_path = joinpath(dir, "report.md")
        @test isfile(report_path)
        report = read(report_path, String)
        @test occursin("# 震源机制反演报告", report)
        @test occursin("30", report)
        @test occursin("NET.STA.Z.P", report)
        @test occursin("Misfit XcorrP", report)
        @test occursin("Total misfit", report)
        @test occursin("N/A", report)
    end
end
