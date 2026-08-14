module Signal

using DSP
using FFTW
using LinearAlgebra
using Statistics

export bandpass_filter!, trim_time_window!, trim_to_polarity_window!
export demean!, detrend!, taper!, preprocess_waveform!
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
    trim_time_window!(obs, gf, dt, arrival_sample, pre_periods, post_periods, band_high)
                      -> (obs_trimmed, gf_trimmed)

Non-symmetric trim: arrival ± pre/post_periods/band_high (period counts dimensionless).
"""
function trim_time_window!(
    obs::Vector{Float64},
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    pre_periods::Float64,
    post_periods::Float64,
    band_high::Float64,
)
    pre_sec = pre_periods / band_high
    post_sec = post_periods / band_high
    pre_n = max(1, round(Int, pre_sec / dt))
    post_n = max(1, round(Int, post_sec / dt))
    start_idx = max(1, arrival_sample - pre_n)
    end_idx = min(length(obs), arrival_sample + post_n)
    return obs[start_idx:end_idx], gf[start_idx:end_idx, :]
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

# 3. Shared Preprocessing Primitives

"""Remove mean from waveform in-place."""
function demean!(wf::Vector{Float64})
    wf .-= sum(wf) / length(wf)
    return wf
end

"""Remove linear trend (least-squares) from waveform in-place."""
function detrend!(wf::Vector{Float64})
    n = length(wf)
    t = collect(1.0:n)
    # least squares: a + b*t
    s_tt = sum(abs2, t) - (sum(t)^2) / n
    s_ty = dot(t, wf) - (sum(t) * sum(wf)) / n
    b = s_ty / s_tt
    a = (sum(wf) - b * sum(t)) / n
    wf .-= (a .+ b .* t)
    return wf
end

"""Apply cosine taper to both ends in-place."""
function taper!(wf::Vector{Float64}; frac::Float64 = 0.05)
    n = length(wf)
    ntap = max(1, round(Int, frac * n))
    for i in 1:ntap
        w = 0.5 * (1 - cos(pi * (i - 1) / ntap))
        wf[i] *= w
        wf[n - i + 1] *= w
    end
    return wf
end

"""
    preprocess_waveform!(wf, dt, low_cut, high_cut; demean=true, detrend=true,
                         taper=true, order=4, do_bandpass=true) -> wf_proc

Full-waveform preprocessing: demean → detrend → taper → bandpass (no trimming).
`do_bandpass=false` skips filtering (Polarity is not frequency-dependent).
"""
function preprocess_waveform!(
    wf::Vector{Float64},
    dt::Float64,
    low_cut::Float64,
    high_cut::Float64;
    demean::Bool = true,
    detrend::Bool = true,
    taper::Bool = true,
    order::Int = 4,
    do_bandpass::Bool = true,
)
    demean && demean!(wf)
    detrend && detrend!(wf)
    taper && taper!(wf)
    do_bandpass && bandpass_filter!(wf, dt, low_cut, high_cut; order = order)
    return wf
end

# Utility functions

"""envelope(x) -> Vector{Float64}; Hilbert envelope (analytic signal magnitude)."""
function envelope(x::AbstractVector{Float64})::Vector{Float64}
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

"""rms_amplitude(x) -> Float64; root-mean-square amplitude of a time series."""
function rms_amplitude(x::AbstractVector{Float64})::Float64
    return sqrt(mean(x .^ 2))
end

end # module
