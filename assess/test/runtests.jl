using Pkg

# Ensure local HDF5IO package is available
hdf5io_path = joinpath(@__DIR__, "..", "..", "shared", "HDF5IO.jl")
if !any(p -> p.name == "HDF5IO", values(Pkg.dependencies()))
    Pkg.develop(path=hdf5io_path)
end

using Test
using HDF5IO
include(joinpath(@__DIR__, "..", "src", "grid_refinement.jl"))
using .GridRefinement

# ═════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════

function make_strategy(; iteration=1, kwargs...)
    defaults = Dict{Symbol,Any}(
        :strike0 => 45.0, :dstrike => 10.0, :nstrike => Int32(3),
        :dip0 => 30.0, :ddip => 8.0, :ndip => Int32(3),
        :rake0 => 90.0, :drake => 8.0, :nrake => Int32(3),
        :depth_indices => Int32[1, 2, 3],
        :freq_indices => Int32[1, 2],
        :xcorr_phase_mask => Int32[1, 1, 1],
        :polarity_station_mask => Int32[1, 1, 1],
        :psr_station_mask => Int32[1, 1, 1],
        :module_weights => [0.5, 0.3, 0.2],
        :best_sdr => [45.0, 30.0, 90.0],
        :best_depth_index => Int32(2),
        :best_misfit => 0.032,
        :iteration => Int32(iteration),
        :converged => Int32(0),
        :convergence_reason => "",
        :freq_accumulated => zeros(Float64, 2, 3),
        :freq_misfit_curve => zeros(Float64, 2, 10),
        :depth_misfit_accumulated => Float64[],
    )
    params = merge(defaults, Dict(kwargs))
    return Strategy(
        params[:strike0], params[:dstrike], params[:nstrike],
        params[:dip0], params[:ddip], params[:ndip],
        params[:rake0], params[:drake], params[:nrake],
        params[:depth_indices], params[:freq_indices],
        params[:xcorr_phase_mask], params[:polarity_station_mask], params[:psr_station_mask],
        params[:module_weights],
        params[:best_sdr], params[:best_depth_index], params[:best_misfit],
        params[:iteration], params[:converged], params[:convergence_reason],
        params[:freq_accumulated], params[:freq_misfit_curve], params[:depth_misfit_accumulated],
    )
end

# ═════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════

@testset "Grid Refinement" begin

    @testset "Grid center matches best trial SDR" begin
        strat = make_strategy()
        best = TrialResult(
            [50.0, 35.0, 85.0],
            Int32(2), Int32(1), 0.025,
            [0.05, 0.03, 0.04],
            [0.03, 0.02],
        )
        result = refine_strategy(strat, best)
        @test result.strike0 == 50.0
        @test result.dip0    == 35.0
        @test result.rake0   == 85.0
    end

    @testset "Step sizes halved correctly" begin
        strat = make_strategy(dstrike=10.0, ddip=8.0, drake=8.0)
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.10, 0.05, 0.08],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.dstrike == 5.0
        @test result.ddip    == 4.0
        @test result.drake   == 4.0
    end

    @testset "Grid always 3×3×3 SDR" begin
        strat = make_strategy(nstrike=Int32(5), ndip=Int32(7), nrake=Int32(11))
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.10, 0.05, 0.08],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.nstrike == 3
        @test result.ndip    == 3
        @test result.nrake   == 3
    end

    @testset "Depth subset within 20% of best" begin
        # depth_misfits: idx 3 (0.03) is best, threshold = 0.036
        # idx 2 (0.08) > 0.036 → excluded
        # idx 1 (0.04) > 0.036 → excluded
        # only idx 3 stays
        strat = make_strategy(depth_indices=Int32[1, 2, 3])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(3), Int32(1), 0.032,
            [0.10, 0.08, 0.03],   # best at idx 3
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test 3 in result.depth_indices
        @test length(result.depth_indices) == 1
    end

    @testset "Frequency subset within 20% of best" begin
        # freq_misfits: idx 2 (0.02) is best, threshold = 0.024
        # idx 1 (0.06) > 0.024 → excluded
        # idx 3 (0.09) > 0.024 → excluded
        # only idx 2 stays
        strat = make_strategy(freq_indices=Int32[1, 2, 3, 4])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(2), 0.032,
            [0.05, 0.04, 0.06],
            [0.10, 0.02, 0.09, 0.12],  # best at idx 2
        )
        result = refine_strategy(strat, best)
        @test 2 in result.freq_indices
        @test length(result.freq_indices) == 1
    end

    @testset "Empty depth subset → single best value" begin
        # best depth = idx 2, misfit = 0.01, threshold = 0.012
        # other depths all > 0.012, so only idx 2
        strat = make_strategy(depth_indices=Int32[1, 2, 3])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [1.0, 0.01, 1.0],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.depth_indices == Int32[2]
    end

    @testset "Empty freq subset → single best value" begin
        # best freq = idx 2, misfit = 0.01, threshold = 0.012
        # other freqs all > 0.012, so only idx 2
        strat = make_strategy(freq_indices=Int32[1, 2, 3])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(2), 0.032,
            [0.05, 0.04, 0.06],
            [1.0, 0.01, 1.0],
        )
        result = refine_strategy(strat, best)
        @test result.freq_indices == Int32[2]
    end

    @testset "Best trial info propagated to output" begin
        strat = make_strategy(iteration=Int32(2))
        best = TrialResult(
            [50.0, 35.0, 85.0],
            Int32(2), Int32(1), 0.025,
            [0.05, 0.03, 0.04],
            [0.03, 0.02],
        )
        result = refine_strategy(strat, best)
        @test result.best_sdr         == [50.0, 35.0, 85.0]
        @test result.best_depth_index == 2
        @test result.best_misfit      == 0.025
        @test result.iteration        == 3
        @test result.converged        == 0
        @test result.convergence_reason == ""
    end

    @testset "Masks and weights preserved through refinement" begin
        strat = make_strategy(
            xcorr_phase_mask=Int32[1, 0, 1, 1, 0],
            polarity_station_mask=Int32[1, 1, 0],
            psr_station_mask=Int32[0, 1, 1],
            module_weights=[0.6, 0.3, 0.1],
        )
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.10, 0.05, 0.08],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.xcorr_phase_mask     == Int32[1, 0, 1, 1, 0]
        @test result.polarity_station_mask == Int32[1, 1, 0]
        @test result.psr_station_mask      == Int32[0, 1, 1]
        @test result.module_weights        == [0.6, 0.3, 0.1]
    end

    @testset "Depth misfit accumulation (element-wise min)" begin
        strat = make_strategy(depth_misfit_accumulated=[0.20, 0.10, 0.30])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.15, 0.08, 0.25],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.depth_misfit_accumulated == [0.15, 0.08, 0.25]
    end

    @testset "First-iteration: empty accumulated → starts fresh" begin
        strat = make_strategy(depth_misfit_accumulated=Float64[])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.15, 0.08, 0.25],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test result.depth_misfit_accumulated == [0.15, 0.08, 0.25]
    end

    @testset "Multiple depths within 20% tolerance" begin
        # best depth idx 2, misfit 0.05, threshold 0.06
        # idx 1: 0.06 <= 0.06 ✓
        # idx 2: 0.05 <= 0.06 ✓  
        # idx 3: 0.055 <= 0.06 ✓
        # idx 4: 0.07 > 0.06 ✗
        # idx 5: 0.10 > 0.06 ✗
        strat = make_strategy(depth_indices=Int32[1, 2, 3, 4, 5])
        best = TrialResult(
            [45.0, 30.0, 90.0],
            Int32(2), Int32(1), 0.032,
            [0.06, 0.05, 0.055, 0.07, 0.10],
            [0.06, 0.04],
        )
        result = refine_strategy(strat, best)
        @test sort(result.depth_indices) == Int32[1, 2, 3]
    end
end

# ═════════════════════════════════════════════════════════
# Operator Prompt Tests
# ═════════════════════════════════════════════════════════

@testset "Operator Prompt" begin

    @testset "Returns true for 'y' (lowercase)" begin
        io_in = IOBuffer("y\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == true
    end

    @testset "Returns true for 'Y' (uppercase)" begin
        io_in = IOBuffer("Y\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == true
    end

    @testset "Returns false for 'N'" begin
        io_in = IOBuffer("N\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == false
    end

    @testset "Returns false for 'n'" begin
        io_in = IOBuffer("n\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == false
    end

    @testset "Returns false for empty input (just Enter)" begin
        io_in = IOBuffer("\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == false
    end

    @testset "Returns false for arbitrary input" begin
        io_in = IOBuffer("maybe\n")
        strat = make_strategy()
        result = prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                                 io_in=io_in, io_out=devnull)
        @test result == false
    end

    @testset "Output contains expected fields" begin
        io_in = IOBuffer("y\n")
        io_out = IOBuffer()
        strat = make_strategy()
        prompt_operator([50.0, 35.0, 85.0], 0.025, strat;
                         io_in=io_in, io_out=io_out)
        output = String(take!(io_out))
        @test occursin("Best SDR", output)
        @test occursin("strike=50.0", output)
        @test occursin("dip=35.0", output)
        @test occursin("rake=85.0", output)
        @test occursin("Misfit=0.025", output)
        @test occursin("Current grid", output)
        @test occursin("Continue?", output)
    end

    @testset "Grid display with varying axes shows ±" begin
        io_in = IOBuffer("y\n")
        io_out = IOBuffer()
        strat = make_strategy()
        prompt_operator([45.0, 30.0, 90.0], 0.032, strat;
                         io_in=io_in, io_out=io_out)
        output = String(take!(io_out))
        @test occursin("strike=45.0±10.0°", output)
        @test occursin("dip=30.0±8.0°", output)
        @test occursin("rake=90.0±8.0°", output)
    end

    @testset "Grid display without varying axes shows no ±" begin
        io_in = IOBuffer("y\n")
        io_out = IOBuffer()
        strat = make_strategy(nstrike=Int32(0), ndip=Int32(0), nrake=Int32(0))
        prompt_operator([45.0, 30.0, 90.0], 0.032, strat;
                         io_in=io_in, io_out=io_out)
        output = String(take!(io_out))
        @test occursin("strike=45.0°", output)
        @test occursin("dip=30.0°", output)
        @test occursin("rake=90.0°", output)
        @test !occursin("±", output)
    end
end