"""Evaluate a compiled objective using a callback for waveform primitives.

The callback receives a primitive `CallNode` and returns a scalar or a
`[entries × trials]` array. Arithmetic is element-wise; `energy` and `rms`
reduce derived values over entries independently for every trial.
"""
evaluate_pipeline(expr::AbstractExpr, resolve_primitive::Function) =
    _evaluate_pipeline(expr, resolve_primitive)

_evaluate_pipeline(expr::LiteralNode, _) = expr.value
_evaluate_pipeline(expr::InputNode, _) =
    throw(ArgumentError("named inputs are not available in pipeline objectives"))
_evaluate_pipeline(expr::WaveformNode, _) =
    throw(ArgumentError("waveform nodes must be consumed by a waveform primitive"))

function _evaluate_pipeline(expr::CallNode, resolve_primitive)
    if _is_direct_waveform_primitive(expr)
        return resolve_primitive(expr)
    end
    args = Tuple(_evaluate_pipeline(arg, resolve_primitive) for arg in expr.args)
    return _evaluate_pipeline_op(expr.op, args)
end

_is_direct_waveform_primitive(expr::CallNode{<:Union{MaxCCOp, LagCCOp}}) = true
_is_direct_waveform_primitive(expr::CallNode{<:Union{AmpScaleOp, SignScaleOp}}) = true
_is_direct_waveform_primitive(expr::CallNode{<:Union{EnergyOp, RMSOp}}) =
    only(expr.args) isa WaveformNode
_is_direct_waveform_primitive(::CallNode) = false

_evaluate_pipeline_op(::AddOp, args) = args[1] .+ args[2]
_evaluate_pipeline_op(::SubtractOp, args) = args[1] .- args[2]
_evaluate_pipeline_op(::MultiplyOp, args) = args[1] .* args[2]
_evaluate_pipeline_op(::DivideOp, args) = args[1] ./ args[2]
_evaluate_pipeline_op(::PowerOp, args) = args[1] .^ args[2]
_evaluate_pipeline_op(::NegateOp, args) = .-args[1]
_evaluate_pipeline_op(::Abs2Op, args) = abs2.(args[1])
_evaluate_pipeline_op(::AbsOp, args) = abs.(args[1])
_evaluate_pipeline_op(::LogOp, args) = log.(args[1])
_evaluate_pipeline_op(::Log10Op, args) = log10.(args[1])
_evaluate_pipeline_op(::SignOp, args) = sign.(args[1])
_evaluate_pipeline_op(::EnergyOp, args) = _pipeline_energy(args[1])
_evaluate_pipeline_op(::RMSOp, args) = sqrt.(_pipeline_energy(args[1]) ./ _pipeline_length(args[1]))

_pipeline_energy(value::Number) = abs2(value)
_pipeline_energy(value::AbstractVector) = sum(abs2, value)
_pipeline_energy(value::AbstractMatrix) = sum(abs2, value; dims = 1)
_pipeline_length(::Number) = 1
_pipeline_length(value::AbstractVector) = length(value)
_pipeline_length(value::AbstractMatrix) = size(value, 1)
