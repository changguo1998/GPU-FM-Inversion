module WaveformPreprocessing

using DSP
using FFTW
using LinearAlgebra
using Statistics

export bandpass_filter!, trim_time_window!, trim_to_polarity_window!
export preprocess_xcorr!, preprocess_polarity!, preprocess_psr!
export envelope, rms_amplitude

# ─────────────────────────────────────────────────────────
# 1. Bandpass Filtering
# ─────────────────────────────────────────────────────────

"""
    bandpass_filter!(x::AbstractVector{Float64}, dt::Float64, low_cut::Float64, high_cut::Float64;
                     order::Int=4)

Apply zero-phase (forward-backward) Butterworth bandpass filter to `x` in-place.

# Arguments
- `x`: signal to filter (modified in-place)
- `dt`: sampling interval (seconds)
- `low_cut`: low-cut corner frequency (Hz)
- `high_cut`: high-cut corner frequency (Hz)
- `order`: Butterworth filter order (default 4)
"""
function bandpass_filter!(x::AbstractVector{Float64}, dt::Float64,
                           low_cut::Float64, high_cut::Float64;
                           order::Int=4)
    fs = 1.0 / dt
    nyquist = fs / 2.0

    # Clamp high_cut below Nyquist to avoid digitalfilter error
    high = min(high_cut, nyquist * 0.999)
    low = max(low_cut, 1e-6)

    if low >= high
        return x  # no feasible passband, leave signal as-is
    end

    responsetype = Bandpass(low, high)
    designmethod = Butterworth(order)
    filt = digitalfilter(responsetype, designmethod; fs=fs)

    n = length(x)
    x[:] = filtfilt(filt, x)
    return x
end

# ─────────────────────────────────────────────────────────
# 2. Time-Window Trimming
# ─────────────────────────────────────────────────────────

"""
    trim_time_window!(obs::Vector{Float64}, gf::Matrix{Float64}, dt::Float64,
                      arrival_sample::Int, window_factor::Float64, band_high::Float64)
                      -> (obs_trimmed::Vector{Float64}, gf_trimmed::Matrix{Float64})

Trim both observed and Green's function waveforms to a time window around the
phase arrival. Returns trimmed copies.

The window is centered on the arrival: from `arrival_sample - N` to `arrival_sample + N`
where `N = round(Int, (window_factor / band_high) / dt)`.

# Arguments
- `obs`: observed waveform (length N_raw)
- `gf`: Green's function matrix (N_raw × 6)
- `dt`: sampling interval (seconds)
- `arrival_sample`: sample index of the P or S arrival (1-indexed)
- `window_factor`: wavelength multiplier for window sizing
- `band_high`: high-cut corner frequency (Hz)

# Returns
- `obs_trimmed`: trimmed observed waveform
- `gf_trimmed`: trimmed GF (trim_window_samples × 6)
"""
function trim_time_window!(obs::Vector{Float64}, gf::Matrix{Float64}, dt::Float64,
                            arrival_sample::Int, window_factor::Float64, band_high::Float64)
    window_seconds = window_factor / band_high
    half_samples = max(1, round(Int, window_seconds / dt))

    n_raw = length(obs)
    start_idx = max(1, arrival_sample - half_samples)
    end_idx = min(n_raw, arrival_sample + half_samples)

    obs_trimmed = obs[start_idx:end_idx]
    gf_trimmed = gf[start_idx:end_idx, :]
    return obs_trimmed, gf_trimmed
end

"""
    trim_to_polarity_window!(gf::Matrix{Float64}, dt::Float64, arrival_sample::Int,
                              t_source::Float64) -> gf_pol::Matrix{Float64}

Trim GF to the polarity window [0, t_source] relative to the arrival,
then sum over time to produce the polarity vector.

Returns gf_pol: (N_polarity_samples × 6) — GF within polarity window.
"""
function trim_to_polarity_window!(gf::Matrix{Float64}, dt::Float64,
                                   arrival_sample::Int, t_source::Float64)
    n_samples = max(1, round(Int, t_source / dt))
    n_raw = size(gf, 1)
    start_idx = arrival_sample
    end_idx = min(n_raw, arrival_sample + n_samples - 1)

    return gf[start_idx:end_idx, :]
end

# ─────────────────────────────────────────────────────────
# 3. Per-Module Preprocessing
# ─────────────────────────────────────────────────────────

"""
    preprocess_xcorr!(obs::Vector{Float64}, gf::Matrix{Float64}, dt::Float64,
                      arrival_sample::Int, low_cut::Float64, high_cut::Float64,
                      window_factor::Float64; filter_order::Int=4)
                      -> (obs_proc::Vector{Float64}, gf_proc::Matrix{Float64},
                          synamp::Matrix{Float64}, obs_norm2::Float64)

Preprocess waveforms for XCorr misfit computation:
1. Bandpass filter both obs and GF columns
2. Trim to time window around arrival
3. Compute synamp = GFᵀ · GF (6×6 Gram matrix)
4. Compute obs_norm² = ‖obs‖²

# Returns
- `obs_proc`: filtered + trimmed observed waveform
- `gf_proc`: filtered + trimmed GF (N_samples × 6)
- `synamp`: 6×6 auto-correlation matrix (GFᵀ · GF)
- `obs_norm2`: squared L2 norm of observed waveform
"""
function preprocess_xcorr!(obs::Vector{Float64}, gf::Matrix{Float64}, dt::Float64,
                            arrival_sample::Int, low_cut::Float64, high_cut::Float64,
                            window_factor::Float64; filter_order::Int=4)
    # Make mutable copies for in-place filtering
    obs_filt = copy(obs)
    n_samples, n_comp = size(gf)
    gf_filt = copy(gf)

    # 1. Bandpass filter
    bandpass_filter!(obs_filt, dt, low_cut, high_cut; order=filter_order)
    for c in 1:n_comp
        col = gf_filt[:, c]
        bandpass_filter!(col, dt, low_cut, high_cut; order=filter_order)
        gf_filt[:, c] = col
    end

    # 2. Trim to time window
    obs_proc, gf_proc = trim_time_window!(
        obs_filt, gf_filt, dt, arrival_sample, window_factor, high_cut
    )

    # 3. Compute synamp = GFᵀ · GF (6×6)
    synamp = gf_proc' * gf_proc

    # 4. Compute obs_norm²
    obs_norm2 = dot(obs_proc, obs_proc)

    return obs_proc, gf_proc, synamp, obs_norm2
end

"""
    preprocess_polarity!(gf::Matrix{Float64}, dt::Float64, arrival_sample::Int,
                          t_source::Float64, obs_polarity::Int8)
                          -> (gf_pol::Matrix{Float64}, obs_pol::Int8)

Preprocess for Polarity misfit computation:
1. Trim GF to polarity window [0, t_source]
2. Pass through observed polarity

# Returns
- `gf_pol`: GF within polarity window (N_polarity_samples × 6)
- `obs_pol`: observed polarity as Int8
"""
function preprocess_polarity!(gf::Matrix{Float64}, dt::Float64,
                               arrival_sample::Int, t_source::Float64,
                               obs_polarity::Int8)
    gf_pol = trim_to_polarity_window!(gf, dt, arrival_sample, t_source)
    return gf_pol, obs_polarity
end

"""
    preprocess_psr!(obs_P::Vector{Float64}, obs_S::Vector{Float64},
                    gf_P::Matrix{Float64}, gf_S::Matrix{Float64},
                    dt::Float64, arrival_P::Int, arrival_S::Int,
                    pre_P_sec::Float64, post_P_sec::Float64,
                    pre_S_sec::Float64, post_S_sec::Float64)
                    -> (amp_P::Matrix{Float64}, amp_S::Matrix{Float64}, obs_psr::Float64)

Preprocess for PSR misfit computation:
1. Compute amp_P = GF_Pᵀ · GF_P (6×6)
2. Compute amp_S = GF_Sᵀ · GF_S (6×6)
3. Compute obs_psr = log10(P_amplitude / S_amplitude)
   where P_amplitude = RMS of obs_P in [arrival_P + pre_P, post_P]
   and   S_amplitude = RMS of obs_S in [arrival_S + pre_S, post_S]

# Returns
- `amp_P`: P-wave amplitude covariance matrix (6×6)
- `amp_S`: S-wave amplitude covariance matrix (6×6)
- `obs_psr`: log10(P/S) observed amplitude ratio
"""
function preprocess_psr!(obs_P::Vector{Float64}, obs_S::Vector{Float64},
                          gf_P::Matrix{Float64}, gf_S::Matrix{Float64},
                          dt::Float64, arrival_P::Int, arrival_S::Int,
                          pre_P_sec::Float64, post_P_sec::Float64,
                          pre_S_sec::Float64, post_S_sec::Float64)
    # Compute amplitude covariance matrices
    amp_P = gf_P' * gf_P
    amp_S = gf_S' * gf_S

    # Trim observed waveforms to P and S windows and compute RMS amplitudes
    pre_P_samples = max(0, round(Int, pre_P_sec / dt))
    post_P_samples = max(1, round(Int, post_P_sec / dt))
    pre_S_samples = max(0, round(Int, pre_S_sec / dt))
    post_S_samples = max(1, round(Int, post_S_sec / dt))

    p_start = max(1, arrival_P - pre_P_samples)
    p_end = min(length(obs_P), arrival_P + post_P_samples)
    s_start = max(1, arrival_S - pre_S_samples)
    s_end = min(length(obs_S), arrival_S + post_S_samples)

    amp_P_obs = rms_amplitude(obs_P[p_start:p_end])
    amp_S_obs = rms_amplitude(obs_S[s_start:s_end])

    obs_psr = if amp_S_obs > 0.0
        log10(amp_P_obs / amp_S_obs)
    else
        0.0
    end

    return amp_P, amp_S, obs_psr
end

# ─────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────

"""
    envelope(x::AbstractVector{Float64}) -> Vector{Float64}

Compute the Hilbert envelope (analytic signal magnitude) of `x`.
"""
function envelope(x::AbstractVector{Float64})
    X = fft(x)
    n = length(x)
    h = zeros(ComplexF64, n)
    if iseven(n)
        h[1] = 1.0
        h[2:(n÷2)] .= 2.0
        h[n÷2+1] = 1.0
    else
        h[1] = 1.0
        h[2:((n+1)÷2)] .= 2.0
    end
    return abs.(ifft(X .* h))
end

"""
    rms_amplitude(x::AbstractVector{Float64}) -> Float64

Compute the root-mean-square amplitude of a time series.
"""
function rms_amplitude(x::AbstractVector{Float64})
    return sqrt(mean(x .^ 2))
end

end # module