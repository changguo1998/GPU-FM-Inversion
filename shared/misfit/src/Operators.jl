struct MaxCCOp <: AbstractOp end
struct LagCCOp <: AbstractOp end
struct EnergyOp <: AbstractOp end
struct AmpScaleOp <: AbstractOp end
struct SignScaleOp <: AbstractOp end
struct RMSOp <: AbstractOp end

"""Return the maximum signed normalized cross-correlation within `maxlag`."""
function maxCC(
    observed::AbstractVector{<:Real},
    synthetic::AbstractVector{<:Real};
    maxlag::Integer = length(observed) - 1,
)
    value, _ = _cc_peak(observed, synthetic, maxlag)
    return value
end

function maxCC(observed::AbstractExpr, synthetic::AbstractExpr; maxlag::Real)
    maxlag >= 0 || throw(ArgumentError("maxlag must be non-negative"))
    return _call(MaxCCOp(), observed, synthetic; maxlag = maxlag)
end

"""Return the sample lag of the maximum signed normalized cross-correlation."""
function lagCC(
    observed::AbstractVector{<:Real},
    synthetic::AbstractVector{<:Real};
    maxlag::Integer = length(observed) - 1,
)
    _, lag = _cc_peak(observed, synthetic, maxlag)
    return lag
end

function lagCC(observed::AbstractExpr, synthetic::AbstractExpr; maxlag::Real)
    maxlag >= 0 || throw(ArgumentError("maxlag must be non-negative"))
    return _call(LagCCOp(), observed, synthetic; maxlag = maxlag)
end

"""Return waveform energy as the sum of squared samples."""
energy(x::AbstractArray{<:Real}) = sum(abs2, x)
energy(x::AbstractExpr) = _call(EnergyOp(), x)

"""Return half the waveform peak-to-peak amplitude."""
function ampScale(x::AbstractArray{<:Real})
    isempty(x) && throw(ArgumentError("ampScale requires a non-empty array"))
    return (maximum(x) - minimum(x)) / 2
end
ampScale(x::AbstractExpr) = _call(AmpScaleOp(), x)

"""Return the sign of the earlier global minimum or maximum."""
function signScale(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("signScale requires a non-empty vector"))
    xmin, imin = findmin(x)
    xmax, imax = findmax(x)
    return sign(imin <= imax ? xmin : xmax)
end
signScale(x::AbstractExpr) = _call(SignScaleOp(), x)

"""Return the root-mean-square waveform amplitude."""
function rms(x::AbstractArray{<:Real})
    isempty(x) && throw(ArgumentError("rms requires a non-empty array"))
    return sqrt(energy(x) / length(x))
end
rms(x::AbstractExpr) = _call(RMSOp(), x)

function _sample_maxlag(kwargs)
    maxlag = kwargs.maxlag
    isinteger(maxlag) ||
        throw(ArgumentError("numeric expression evaluation requires integer maxlag"))
    return Int(maxlag)
end

_evaluate_op(::MaxCCOp, args, kwargs) = maxCC(args...; maxlag = _sample_maxlag(kwargs))
_evaluate_op(::LagCCOp, args, kwargs) = lagCC(args...; maxlag = _sample_maxlag(kwargs))
_evaluate_op(::EnergyOp, args, _) = energy(args[1])
_evaluate_op(::AmpScaleOp, args, _) = ampScale(args[1])
_evaluate_op(::SignScaleOp, args, _) = signScale(args[1])
_evaluate_op(::RMSOp, args, _) = rms(args[1])

_op_name(::MaxCCOp) = "max_cc"
_op_name(::LagCCOp) = "lag_cc"
_op_name(::EnergyOp) = "energy"
_op_name(::AmpScaleOp) = "amp_scale"
_op_name(::SignScaleOp) = "sign_scale"
_op_name(::RMSOp) = "rms"

_op_from_val(::Val{:max_cc}) = MaxCCOp()
_op_from_val(::Val{:lag_cc}) = LagCCOp()
_op_from_val(::Val{:energy}) = EnergyOp()
_op_from_val(::Val{:amp_scale}) = AmpScaleOp()
_op_from_val(::Val{:sign_scale}) = SignScaleOp()
_op_from_val(::Val{:rms}) = RMSOp()

function _cc_peak(
    observed::AbstractVector{<:Real},
    synthetic::AbstractVector{<:Real},
    maxlag::Integer,
)
    isempty(observed) && throw(ArgumentError("cross-correlation requires non-empty vectors"))
    length(observed) == length(synthetic) ||
        throw(ArgumentError("cross-correlation vectors must have equal length"))
    maxlag >= 0 || throw(ArgumentError("maxlag must be non-negative"))

    observed_energy = energy(observed)
    observed_energy <= 0 && return 0.0, 0

    n = length(observed)
    effective_maxlag = min(maxlag, n - 1)
    best_cc = 0.0
    best_lag = 0
    found = false
    for lag in (-effective_maxlag):effective_maxlag
        dot_value = 0.0
        synthetic_energy = 0.0
        for i in eachindex(observed)
            j = i - lag
            if firstindex(synthetic) <= j <= lastindex(synthetic)
                value = synthetic[j]
                dot_value += observed[i] * value
                synthetic_energy += value * value
            end
        end
        synthetic_energy <= 0 && continue
        cc = dot_value / sqrt(observed_energy * synthetic_energy)
        if !found || cc > best_cc
            best_cc = cc
            best_lag = lag
            found = true
        end
    end
    return best_cc, best_lag
end
