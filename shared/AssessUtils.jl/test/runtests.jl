using Test
using AssessUtils

@testset "Misfit Aggregator" begin
    @testset "hand-computed example" begin
        # 2 phases × 3 trials
        xcorr = [
            0.1  0.2  0.3;
            0.4  0.5  0.6
        ]
        # 2 stations × 3 trials
        polarity = [
            0.0  1.0  0.0;
            1.0  0.0  1.0
        ]
        psr = [
            0.01  0.02  0.03;
            0.04  0.05  0.06
        ]

        mask_xc = [true, true]   # both phases active
        mask_pol = [true, true]  # both stations active
        mask_psr = [true, true]  # both stations active
        weights = [1.0, 1.0, 1.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # XCorr: sum over 2 phases → [0.5, 0.7, 0.9]
        @test per_module[:xcorr] ≈ [0.5, 0.7, 0.9]
        # Polarity: sum over 2 stations → [1.0, 1.0, 1.0]
        @test per_module[:polarity] ≈ [1.0, 1.0, 1.0]
        # PSR: sum over 2 stations → [0.05, 0.07, 0.09]
        @test per_module[:psr] ≈ [0.05, 0.07, 0.09]
        # Total: sum of all → [1.55, 1.77, 1.99]
        @test total ≈ [1.55, 1.77, 1.99]
        # Best: trial 1 has minimum total (1.55)
        @test best_idx == 1
    end

    @testset "masked phases do not affect totals" begin
        # 3 phases × 2 trials
        xcorr = [
            0.2  0.3;   # phase 1
            9.9  9.9;   # phase 2 (will be masked)
            0.4  0.5    # phase 3
        ]
        polarity = [0.0  1.0]  # 1 station × 2 trials
        psr = [0.01  0.02]     # 1 station × 2 trials

        # Mask out phase 2
        mask_xc = [true, false, true]
        mask_pol = [true]
        mask_psr = [true]
        weights = [1.0, 1.0, 1.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # XCorr: only phases 1 + 3 → [0.6, 0.8] (phase 2's 9.9 ignored)
        @test per_module[:xcorr] ≈ [0.6, 0.8]
    end

    @testset "masked stations do not affect totals" begin
        xcorr = [0.1 0.2]  # 1 phase × 2 trials
        # 2 stations × 2 trials, station 2 masked
        polarity = [
            0.0  1.0;   # station 1
            9.9  9.9    # station 2 (masked)
        ]
        psr = [
            0.01  0.02;
            9.9   9.9    # station 2 (masked)
        ]

        mask_xc = [true]
        mask_pol = [true, false]
        mask_psr = [true, false]
        weights = [1.0, 1.0, 1.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # Polarity: only station 1 → [0.0, 1.0]
        @test per_module[:polarity] ≈ [0.0, 1.0]
        # PSR: only station 1 → [0.01, 0.02]
        @test per_module[:psr] ≈ [0.01, 0.02]
    end

    @testset "weight=0: module contributes nothing" begin
        xcorr = [1.0  2.0]
        polarity = [10.0  20.0]
        psr = [100.0  200.0]

        mask_xc = [true]
        mask_pol = [true]
        mask_psr = [true]
        weights = [0.0, 0.0, 0.0]  # all zero

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # Total should be all zeros
        @test total ≈ [0.0, 0.0]
        @test best_idx == 1  # all-zero tie → first trial

        # Single module weight zero
        weights2 = [1.0, 0.0, 1.0]
        total2, best2, per2 = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights2,
        )
        # Only xcorr and psr contribute
        @test total2 ≈ [101.0, 202.0]
    end

    @testset "mixed NaN in data: non-NaN values still counted" begin
        # Phase 1 has NaN for trial 2 — only phase 0 counts for trial 2
        xcorr = [
            1.0  NaN;   # phase 0: valid for trial 1, NaN for trial 2
            2.0  3.0     # phase 1: both valid
        ]
        polarity = [0.0  0.0]
        psr = [0.0  0.0]

        mask_xc = [true, true]
        mask_pol = [true]
        mask_psr = [true]
        weights = [1.0, 0.0, 0.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # XCorr trial 1: 1.0 + 2.0 = 3.0, trial 2: NaN skipped → 3.0
        @test per_module[:xcorr] ≈ [3.0, 3.0]
    end

    @testset "all-NaN: errors" begin
        xcorr = [NaN  NaN]
        polarity = [NaN  NaN]
        psr = [NaN  NaN]

        mask_xc = [true]
        mask_pol = [true]
        mask_psr = [true]
        weights = [1.0, 1.0, 1.0]

        @test_throws ErrorException aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )
    end

    @testset "all masked: produces NaN per-trial (no active rows)" begin
        xcorr = [1.0  2.0; 3.0  4.0]
        polarity = [0.5  0.5]
        psr = [0.1  0.1]

        # All xcorr phases masked
        mask_xc = [false, false]
        mask_pol = [true]
        mask_psr = [true]
        weights = [1.0, 1.0, 1.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        # XCorr should be NaN for all trials, polarity + psr still valid
        @test all(isnan, per_module[:xcorr])
        @test !any(isnan, per_module[:polarity])
        @test !any(isnan, per_module[:psr])
        # Total should only include polarity + psr
        @test total ≈ [0.6, 0.6]
    end

    @testset "single phase, single trial" begin
        xcorr = [0.5;;]           # 1×1 matrix
        polarity = [0.0;;]         # 1×1 matrix
        psr = [0.01;;]             # 1×1 matrix

        mask_xc = [true]
        mask_pol = [true]
        mask_psr = [true]
        weights = [1.0, 0.5, 2.0]

        total, best_idx, per_module = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )

        @test per_module[:xcorr] ≈ [0.5]
        @test per_module[:polarity] ≈ [0.0]
        @test per_module[:psr] ≈ [0.01]
        @test total ≈ [0.52]  # 1*0.5 + 0.5*0 + 2*0.01 = 0.5 + 0.02 = 0.52
        @test best_idx == 1
    end

    @testset "weighted module dominance" begin
        xcorr = [0.1  0.9]
        polarity = [0.0  1.0]
        psr = [0.001  0.002]

        mask_xc = [true]
        mask_pol = [true]
        mask_psr = [true]

        # Heavily weight polarity → trial 1 wins (lower polarity)
        weights = [1.0, 100.0, 1.0]
        total, best_idx, _ = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )
        @test best_idx == 1  # trial 1: polarity 0 → 0*100=0, trial 2: 1*100=100

        # Heavily weight xcorr → trial 1 wins (lower xcorr)
        weights = [100.0, 1.0, 1.0]
        total, best_idx, _ = aggregate_misfits(
            xcorr, polarity, psr,
            mask_xc, mask_pol, mask_psr, weights,
        )
        @test best_idx == 1  # trial 1: xcorr 0.1*100=10, trial 2: 0.9*100=90
    end
end

println("All AssessUtils tests passed!")