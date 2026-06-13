using Test
using LinearAlgebra
using DSP
using Statistics

# Add the input/src to LOAD_PATH so we can include the module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "preprocess.jl"))
using .WaveformPreprocessing

# ─────────────────────────────────────────────────────────
# Test 1: Bandpass Filtering
# ─────────────────────────────────────────────────────────
@testset "Bandpass filter" begin
    dt = 0.01        # 100 Hz sampling
    fs = 1.0 / dt
    t = 0:dt:10.0
    t = collect(t)

    # Generate 5 Hz sine with noise
    f_signal = 5.0
    x = sin.(2π * f_signal * t) + 0.2 * randn(length(t))

    # Apply bandpass [3, 8] Hz
    x_filt = copy(x)
    bandpass_filter!(x_filt, dt, 3.0, 8.0; order=4)

    # Check that output is not identical to input (filtering happened)
    @test norm(x_filt - x) > 0.0

    # Check no NaN produced
    @test !any(isnan, x_filt)

    # Check length preserved
    @test length(x_filt) == length(x)

    # Edge case: low_cut >= high_cut should return unchanged
    x_edge = copy(x)
    bandpass_filter!(x_edge, dt, 10.0, 3.0; order=4)
    @test x_edge ≈ x

    @info "Bandpass filter tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 2: Time-Window Trimming
# ─────────────────────────────────────────────────────────
@testset "Time-window trimming" begin
    dt = 0.02         # 50 Hz
    n_raw = 1000
    arrival_sample = 400
    window_factor = 2.0
    band_high = 10.0  # Hz

    obs = collect(1.0:n_raw)
    gf = randn(n_raw, 6)

    obs_trim, gf_trim = trim_time_window!(
        obs, gf, dt, arrival_sample, window_factor, band_high
    )

    # Window size: window_seconds = 2.0 / 10.0 = 0.2 s
    # half_samples = round(Int, 0.2 / 0.02) = 10
    window_seconds = window_factor / band_high
    half_samples = max(1, round(Int, window_seconds / dt))
    @test half_samples == 10

    expected_start = arrival_sample - half_samples  # 390
    expected_end = arrival_sample + half_samples    # 410
    expected_len = expected_end - expected_start + 1

    @test length(obs_trim) == expected_len       # 21
    @test size(gf_trim, 1) == expected_len
    @test size(gf_trim, 2) == 6
    @test obs_trim[1] ≈ obs[expected_start]       # First element matches

    # Test boundary clamping: arrival near start
    obs2, gf2 = trim_time_window!(obs, gf, dt, 5, window_factor, band_high)
    @test length(obs2) >= 1
    @test obs2[1] ≈ obs[1]

    @info "Time-window trimming tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 3: Polarity window trimming
# ─────────────────────────────────────────────────────────
@testset "Polarity window" begin
    dt = 0.01
    n_raw = 500
    arrival_sample = 200
    t_source = 1.0  # 1 second

    gf = randn(n_raw, 6)

    gf_pol = trim_to_polarity_window!(gf, dt, arrival_sample, t_source)

    expected_samples = round(Int, t_source / dt)  # 100
    @test size(gf_pol, 1) == expected_samples
    @test size(gf_pol, 2) == 6
    @test gf_pol[1, :] ≈ gf[arrival_sample, :]     # Starts at arrival

    @info "Polarity window tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 4: XCorr preprocessing — synamp identity
# ─────────────────────────────────────────────────────────
@testset "XCorr preprocessing - synamp identity" begin
    dt = 0.02
    n_raw = 600
    arrival_sample = 300
    low_cut = 1.0
    high_cut = 8.0
    window_factor = 2.0
    filter_order = 4

    # Generate a simple synthetic signal
    t = (0:n_raw-1) * dt
    obs = sin.(2π * 3.0 * t) .+ 0.1 * randn(n_raw)
    gf = randn(n_raw, 6)

    obs_proc, gf_proc, synamp, obs_norm2 = preprocess_xcorr!(
        obs, gf, dt, arrival_sample, low_cut, high_cut, window_factor;
        filter_order=filter_order
    )

    # Check output shapes
    @test length(obs_proc) == size(gf_proc, 1)
    @test size(gf_proc, 2) == 6
    @test size(synamp) == (6, 6)

    # Symmetry: synamp should be symmetric
    @test synamp ≈ synamp'

    # obs_norm² should equal dot(obs_proc, obs_proc)
    @test obs_norm2 ≈ dot(obs_proc, obs_proc)

    # Synamp identity: mᵀ · synamp · m ≈ ‖GF · m‖² for random m
    for _ in 1:10
        m = randn(6)
        lhs = m' * synamp * m
        rhs = sum(abs2, gf_proc * m)
        @test lhs ≈ rhs  atol=1e-8
    end

    @info "XCorr preprocessing + synamp identity tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 5: Polarity preprocessing
# ─────────────────────────────────────────────────────────
@testset "Polarity preprocessing" begin
    dt = 0.01
    n_raw = 500
    arrival_sample = 200
    t_source = 0.5
    obs_polarity = Int8(1)

    gf = randn(n_raw, 6)

    gf_pol, obs_pol = preprocess_polarity!(
        gf, dt, arrival_sample, t_source, obs_polarity
    )

    expected_samples = round(Int, t_source / dt)  # 50
    @test size(gf_pol, 1) == expected_samples
    @test size(gf_pol, 2) == 6
    @test obs_pol == obs_polarity

    @info "Polarity preprocessing tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 6: PSR preprocessing
# ─────────────────────────────────────────────────────────
@testset "PSR preprocessing" begin
    dt = 0.01
    n_raw = 1000
    arrival_P = 300
    arrival_S = 500

    # Create P-wave with amplitude 2.0 and S-wave with amplitude 0.5
    t = (0:n_raw-1) * dt
    obs_P = 2.0 * sin.(2π * 3.0 * t) .+ 0.05 * randn(n_raw)
    obs_S = 0.5 * sin.(2π * 3.0 * t) .+ 0.05 * randn(n_raw)

    gf_P = randn(n_raw, 6)
    gf_S = randn(n_raw, 6)

    pre_P_sec = 0.2
    post_P_sec = 1.0
    pre_S_sec = 0.2
    post_S_sec = 1.0

    amp_P, amp_S, obs_psr = preprocess_psr!(
        obs_P, obs_S, gf_P, gf_S, dt,
        arrival_P, arrival_S,
        pre_P_sec, post_P_sec,
        pre_S_sec, post_S_sec
    )

    # Check shapes
    @test size(amp_P) == (6, 6)
    @test size(amp_S) == (6, 6)

    # Symmetry
    @test amp_P ≈ amp_P'
    @test amp_S ≈ amp_S'

    # obs_psr = log10(P/S). With amp_P ≈ 2.0 and amp_S ≈ 0.5,
    # ratio ≈ 4.0, log10 ≈ 0.602
    @test obs_psr > 0.0
    @test obs_psr < 1.0

    # Known ratio test: create exact amplitude P=4.0, S=1.0
    obs_P2 = 4.0 * sin.(2π * 3.0 * t)
    obs_S2 = 1.0 * sin.(2π * 3.0 * t)
    _, _, obs_psr2 = preprocess_psr!(
        obs_P2, obs_S2, gf_P, gf_S, dt,
        arrival_P, arrival_S,
        pre_P_sec, post_P_sec,
        pre_S_sec, post_S_sec
    )
    @test obs_psr2 ≈ log10(4.0 / 1.0) atol=0.01

    # Zero S amplitude edge case
    obs_S_zero = zeros(n_raw)
    _, _, obs_psr_zero = preprocess_psr!(
        obs_P2, obs_S_zero, gf_P, gf_S, dt,
        arrival_P, arrival_S,
        pre_P_sec, post_P_sec,
        pre_S_sec, post_S_sec
    )
    @test obs_psr_zero == 0.0

    @info "PSR preprocessing tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 7: RMS amplitude utility
# ─────────────────────────────────────────────────────────
@testset "RMS amplitude" begin
    @test rms_amplitude([1.0, 2.0, 3.0]) ≈ sqrt((1 + 4 + 9) / 3)
    @test rms_amplitude([0.0, 0.0]) ≈ 0.0
    @test rms_amplitude([5.0]) ≈ 5.0
    @info "RMS amplitude tests passed"
end

# ─────────────────────────────────────────────────────────
# Test 8: Envelope utility
# ─────────────────────────────────────────────────────────
@testset "Envelope" begin
    dt = 0.01
    t = collect(0:dt:5.0)
    x = sin.(2π * 2.0 * t)
    env = envelope(x)

    @test length(env) == length(x)
    # Envelope of a pure sine should be close to 1.0 everywhere
    @test all(env .> 0.0)
    @test mean(env[10:end-10]) ≈ 1.0 atol=0.2

    @info "Envelope tests passed"
end

println("\nAll waveform preprocessing tests passed!")