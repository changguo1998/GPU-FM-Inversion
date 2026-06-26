module Signal

using DSP
using FFTW
using LinearAlgebra
using Statistics

export bandpass_filter!, trim_time_window!, trim_to_polarity_window!
export preprocess_xcorr!, preprocess_polarity!, preprocess_psr!
export envelope, rms_amplitude

# 1. Bandpass Filtering

"""
    bandpass_filter!(x::AbstractVector{Float64}, dt::Float64, low_cut::Float64, high_cut::Float64;
                     order::Int=4)

Apply zero-phase (forward-backward) Butterworth bandpass filter to `x` in-place.
"""
function bandpass_filter!(
    x::AbstractVector{Float64},
    dt::Float64,
    low_cut::Float64,
    high_cut::Float64;
    order::Int = 4,
)
    fs = 1.0 / dt
    nyquist = fs / 2.0

    high = min(high_cut, nyquist * 0.999)
    low = max(low_cut, 1e-6)

    if low >= high
        return x
    end

    responsetype = Bandpass(low, high)
    designmethod = Butterworth(order)
    filt = digitalfilter(responsetype, designmethod; fs = fs)

    n = length(x)
    x[:] = filtfilt(filt, x)
    return x
end

# 2. Time-Window Trimming

"""
    trim_time_window!(obs::Vector{Float64}, gf::Matrix{Float64}, dt::Float64,
                      arrival_sample::Int, window_factor::Float64, band_high::Float64)
                      -> (obs_trimmed, gf_trimmed)
"""
function trim_time_window!(
    obs::Vector{Float64},
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    window_factor::Float64,
    band_high::Float64,
)
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
                              t_source::Float64) -> gf_pol
"""
function trim_to_polarity_window!(
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    t_source::Float64,
)
    n_samples = max(1, round(Int, t_source / dt))
    n_raw = size(gf, 1)
    start_idx = arrival_sample
    end_idx = min(n_raw, arrival_sample + n_samples - 1)

    return gf[start_idx:end_idx, :]
end

# 3. Per-Module Preprocessing

"""
    preprocess_xcorr!(obs, gf, dt, arrival_sample, low_cut, high_cut, window_factor;
                      filter_order=4)
                      -> (obs_proc, gf_proc, synamp, obs_norm2)
"""
function preprocess_xcorr!(
    obs::Vector{Float64},
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    low_cut::Float64,
    high_cut::Float64,
    window_factor::Float64;
    filter_order::Int = 4,
)
    obs_filt = copy(obs)
    n_samples, n_comp = size(gf)
    gf_filt = copy(gf)

    bandpass_filter!(obs_filt, dt, low_cut, high_cut; order = filter_order)
    for c in 1:n_comp
        col = gf_filt[:, c]
        bandpass_filter!(col, dt, low_cut, high_cut; order = filter_order)
        gf_filt[:, c] = col
    end

    obs_proc, gf_proc =
        trim_time_window!(obs_filt, gf_filt, dt, arrival_sample, window_factor, high_cut)

    synamp = gf_proc' * gf_proc
    obs_norm2 = dot(obs_proc, obs_proc)

    return obs_proc, gf_proc, synamp, obs_norm2
end

"""
    preprocess_polarity!(gf, dt, arrival_sample, t_source, obs_polarity)
                         -> (gf_pol, obs_pol)
"""
function preprocess_polarity!(
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    t_source::Float64,
    obs_polarity::Int8,
)
    gf_pol = trim_to_polarity_window!(gf, dt, arrival_sample, t_source)
    obs_pol_float = if obs_polarity == Int8(-128)
        NaN
    else
        Float64(obs_polarity)
    end
    return gf_pol, obs_pol_float
end

"""
    preprocess_psr!(obs_P, obs_S, gf_P, gf_S, dt, arrival_P, arrival_S,
                    pre_P_sec, post_P_sec, pre_S_sec, post_S_sec)
                    -> (amp_P, amp_S, obs_psr)
"""
function preprocess_psr!(
    obs_P::Vector{Float64},
    obs_S::Vector{Float64},
    gf_P::Matrix{Float64},
    gf_S::Matrix{Float64},
    dt::Float64,
    arrival_P::Int,
    arrival_S::Int,
    pre_P_sec::Float64,
    post_P_sec::Float64,
    pre_S_sec::Float64,
    post_S_sec::Float64,
)
    amp_P = gf_P' * gf_P
    amp_S = gf_S' * gf_S

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

# Utility functions

"""
    envelope(x) -> Vector{Float64}

Compute the Hilbert envelope (analytic signal magnitude) of `x`.
"""
function envelope(x::AbstractVector{Float64})
    X = fft(x)
    n = length(x)
    h = zeros(ComplexF64, n)
    if iseven(n)
        h[1] = 1.0
        h[2:(n ÷ 2)] .= 2.0
        h[n ÷ 2 + 1] = 1.0
    else
        h[1] = 1.0
        h[2:((n + 1) ÷ 2)] .= 2.0
    end
    return abs.(ifft(X .* h))
end

"""
    rms_amplitude(x) -> Float64

Compute the root-mean-square amplitude of a time series.
"""
function rms_amplitude(x::AbstractVector{Float64})
    return sqrt(mean(x .^ 2))
end

end # module
