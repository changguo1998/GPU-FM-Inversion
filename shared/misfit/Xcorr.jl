# XCorr misfit plugin (template)
#
# Included inside Config.{name} (dynamically created inner module).
# Template for cross-correlation misfit — instantiated per phase via
# `Config.use_misfit!(:XcorrP, from = :Xcorr)`.
#
# Config stubs (user must override):
#   trim()     — time window [pre, post] seconds relative to arrival
#   maxlag_factor(), filter_order(), select_threshold(), deselect_threshold()

export trim, maxlag_factor, filter_order
export select_threshold, deselect_threshold, preprocess

# -- Config namespace (user must override) --

function trim()
    throw(Config.ConfigError("Xcorr.trim", "-> Vector{Float64}  (e.g. return [-2.0, 5.0])"))
end

function maxlag_factor()
    throw(Config.ConfigError("Xcorr.maxlag_factor", "-> Float64"))
end

function filter_order()
    throw(Config.ConfigError("Xcorr.filter_order", "-> Int"))
end

function select_threshold()
    throw(Config.ConfigError("Xcorr.select_threshold", "-> Float64"))
end

function deselect_threshold()
    throw(Config.ConfigError("Xcorr.deselect_threshold", "-> Float64"))
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))

"""
    preprocess(obs, gf, dt, arrival_sample, low_cut, high_cut, window_factor;
               filter_order=4) -> (obs_proc, gf_proc, synamp, obs_norm2)

Bandpass filter + time-window trim for cross-correlation misfit.
"""
function preprocess(
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
    gf_filt = copy(gf)

    _Signal.bandpass_filter!(obs_filt, dt, low_cut, high_cut; order = filter_order)
    for c in 1:size(gf, 2)
        col = gf_filt[:, c]
        _Signal.bandpass_filter!(col, dt, low_cut, high_cut; order = filter_order)
        gf_filt[:, c] = col
    end

    obs_proc, gf_proc =
        _Signal.trim_time_window!(obs_filt, gf_filt, dt, arrival_sample, window_factor, high_cut)

    synamp = gf_proc' * gf_proc
    obs_norm2 = sum(obs_proc .^ 2)

    return obs_proc, gf_proc, synamp, obs_norm2
end
