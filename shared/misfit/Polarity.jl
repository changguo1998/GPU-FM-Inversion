# Polarity misfit plugin
#
# Included inside Config.Polarity (dynamically created inner module).
# Config function stubs + preprocessing logic.

export trim, preprocess

# -- Config namespace (user must override) --

function trim()
    throw(Config.ConfigError("Polarity.trim", "-> Vector{Float64}"))
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))

"""
    preprocess(gf, dt, arrival_sample, t_source, obs_polarity) -> (gf_pol, obs_pol)

Trim GF to polarity window.
"""
function preprocess(
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    t_source::Float64,
    obs_polarity::Int8,
)
    gf_pol = _Signal.trim_to_polarity_window!(gf, dt, arrival_sample, t_source)
    obs_pol_float = if obs_polarity == Int8(-128)
        NaN
    else
        Float64(obs_polarity)
    end
    return gf_pol, obs_pol_float
end
