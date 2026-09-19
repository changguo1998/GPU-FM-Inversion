"""Base type for target-function expression nodes."""
abstract type AbstractExpr end

abstract type AbstractOp end

struct Phase{Name} end

const P = Phase{:P}()
const S = Phase{:S}()

struct InputNode <: AbstractExpr
    name::Symbol
end

struct LiteralNode{T <: Number} <: AbstractExpr
    value::T
end

struct CallNode{O <: AbstractOp, A <: Tuple, K <: NamedTuple} <: AbstractExpr
    op::O
    args::A
    kwargs::K
end

struct WaveformNode <: AbstractExpr
    role::Symbol
    phase::Symbol
    band::NTuple{2, Float64}
    window::NTuple{2, Float64}
    channel::Union{Nothing, String}
    filter_order::Int
end

"""Create a named input node for a target-function expression."""
input(name::Symbol) = InputNode(name)

"""Create an observed-waveform source node."""
function observed(
    phase::Phase{Name};
    band,
    window,
    channel::Union{Nothing, String} = nothing,
    filter_order::Integer = 4,
) where {Name}
    return _waveform(:observed, Name, band, window, channel, filter_order)
end

"""Create a synthetic-waveform source node."""
function synthetic(
    phase::Phase{Name};
    band,
    window,
    channel::Union{Nothing, String} = nothing,
    filter_order::Integer = 4,
) where {Name}
    return _waveform(:synthetic, Name, band, window, channel, filter_order)
end

function _waveform(role, phase, band, window, channel, filter_order)
    length(band) == 2 || throw(ArgumentError("band must contain low and high frequencies"))
    length(window) == 2 || throw(ArgumentError("window must contain start and end periods"))
    band_value = (Float64(band[1]), Float64(band[2]))
    window_value = (Float64(window[1]), Float64(window[2]))
    0.0 <= band_value[1] < band_value[2] || throw(ArgumentError("invalid frequency band"))
    window_value[1] < window_value[2] || throw(ArgumentError("invalid waveform window"))
    filter_order > 0 || throw(ArgumentError("filter_order must be positive"))
    return WaveformNode(role, phase, band_value, window_value, channel, Int(filter_order))
end

"""Evaluate an expression with values supplied by input name."""
evaluate(expr::AbstractExpr, inputs::AbstractDict) = _evaluate(expr, inputs)

"""Encode an expression as a nested dictionary suitable for HDF5 storage."""
encode_expression(expr::AbstractExpr) = _encode_expression(expr)

"""Decode an expression from its nested dictionary representation."""
decode_expression(data::AbstractDict) = _decode_expression(data)

struct AddOp <: AbstractOp end
struct SubtractOp <: AbstractOp end
struct MultiplyOp <: AbstractOp end
struct DivideOp <: AbstractOp end
struct PowerOp <: AbstractOp end
struct NegateOp <: AbstractOp end
struct Abs2Op <: AbstractOp end
struct AbsOp <: AbstractOp end
struct LogOp <: AbstractOp end
struct Log10Op <: AbstractOp end
struct SignOp <: AbstractOp end

_expr(x::AbstractExpr) = x
_expr(x::Number) = LiteralNode(x)
_call(op::AbstractOp, args...; kwargs...) =
    CallNode(op, Tuple(_expr(arg) for arg in args), (; kwargs...))

Base.:+(left::AbstractExpr, right::Union{AbstractExpr, Number}) = _call(AddOp(), left, right)
Base.:+(left::Number, right::AbstractExpr) = _call(AddOp(), left, right)
Base.:-(left::AbstractExpr, right::Union{AbstractExpr, Number}) = _call(SubtractOp(), left, right)
Base.:-(left::Number, right::AbstractExpr) = _call(SubtractOp(), left, right)
Base.:*(left::AbstractExpr, right::Union{AbstractExpr, Number}) = _call(MultiplyOp(), left, right)
Base.:*(left::Number, right::AbstractExpr) = _call(MultiplyOp(), left, right)
Base.:/(left::AbstractExpr, right::Union{AbstractExpr, Number}) = _call(DivideOp(), left, right)
Base.:/(left::Number, right::AbstractExpr) = _call(DivideOp(), left, right)
Base.:^(left::AbstractExpr, right::Union{AbstractExpr, Number}) = _call(PowerOp(), left, right)
Base.:^(left::Number, right::AbstractExpr) = _call(PowerOp(), left, right)
Base.:-(arg::AbstractExpr) = _call(NegateOp(), arg)
Base.abs2(arg::AbstractExpr) = _call(Abs2Op(), arg)
Base.abs(arg::AbstractExpr) = _call(AbsOp(), arg)
Base.log(arg::AbstractExpr) = _call(LogOp(), arg)
Base.log10(arg::AbstractExpr) = _call(Log10Op(), arg)
Base.sign(arg::AbstractExpr) = _call(SignOp(), arg)

_evaluate(expr::InputNode, inputs) = inputs[expr.name]
_waveform_key(expr::WaveformNode) =
    (expr.role, expr.phase, expr.band, expr.window, expr.channel, expr.filter_order)
_evaluate(expr::WaveformNode, inputs) = inputs[_waveform_key(expr)]
_evaluate(expr::LiteralNode, _) = expr.value
function _evaluate(expr::CallNode, inputs)
    args = Tuple(_evaluate(arg, inputs) for arg in expr.args)
    return _evaluate_op(expr.op, args, expr.kwargs)
end

_evaluate_op(::AddOp, args, _) = args[1] + args[2]
_evaluate_op(::SubtractOp, args, _) = args[1] - args[2]
_evaluate_op(::MultiplyOp, args, _) = args[1] * args[2]
_evaluate_op(::DivideOp, args, _) = args[1] / args[2]
_evaluate_op(::PowerOp, args, _) = args[1]^args[2]
_evaluate_op(::NegateOp, args, _) = -args[1]
_evaluate_op(::Abs2Op, args, _) = abs2(args[1])
_evaluate_op(::AbsOp, args, _) = abs(args[1])
_evaluate_op(::LogOp, args, _) = log(args[1])
_evaluate_op(::Log10Op, args, _) = log10(args[1])
_evaluate_op(::SignOp, args, _) = sign(args[1])

_op_name(::AddOp) = "add"
_op_name(::SubtractOp) = "subtract"
_op_name(::MultiplyOp) = "multiply"
_op_name(::DivideOp) = "divide"
_op_name(::PowerOp) = "power"
_op_name(::NegateOp) = "negate"
_op_name(::Abs2Op) = "abs2"
_op_name(::AbsOp) = "abs"
_op_name(::LogOp) = "log"
_op_name(::Log10Op) = "log10"
_op_name(::SignOp) = "sign"

_op_from_val(::Val{:add}) = AddOp()
_op_from_val(::Val{:subtract}) = SubtractOp()
_op_from_val(::Val{:multiply}) = MultiplyOp()
_op_from_val(::Val{:divide}) = DivideOp()
_op_from_val(::Val{:power}) = PowerOp()
_op_from_val(::Val{:negate}) = NegateOp()
_op_from_val(::Val{:abs2}) = Abs2Op()
_op_from_val(::Val{:abs}) = AbsOp()
_op_from_val(::Val{:log}) = LogOp()
_op_from_val(::Val{:log10}) = Log10Op()
_op_from_val(::Val{:sign}) = SignOp()
_op_from_val(::Val{name}) where {name} = throw(ArgumentError("unknown expression operator: $name"))

_encode_expression(expr::InputNode) =
    Dict{String, Any}("kind" => "input", "name" => string(expr.name))
_encode_expression(expr::WaveformNode) = Dict{String, Any}(
    "kind" => "waveform",
    "role" => string(expr.role),
    "phase" => string(expr.phase),
    "band" => collect(expr.band),
    "window" => collect(expr.window),
    "channel" => something(expr.channel, ""),
    "filter_order" => Int32(expr.filter_order),
)
_encode_expression(expr::LiteralNode) =
    Dict{String, Any}("kind" => "literal", "value" => expr.value)
function _encode_expression(expr::CallNode)
    args =
        Dict{String, Any}(string(i) => _encode_expression(arg) for (i, arg) in enumerate(expr.args))
    kwargs = Dict{String, Any}(string(name) => value for (name, value) in pairs(expr.kwargs))
    return Dict{String, Any}(
        "kind" => "call",
        "op" => _op_name(expr.op),
        "args" => args,
        "kwargs" => kwargs,
    )
end

function _decode_expression(data::AbstractDict)
    kind = String(data["kind"])
    kind == "input" && return InputNode(Symbol(data["name"]))
    if kind == "waveform"
        channel = String(data["channel"])
        return WaveformNode(
            Symbol(data["role"]),
            Symbol(data["phase"]),
            Tuple(Float64.(data["band"])),
            Tuple(Float64.(data["window"])),
            isempty(channel) ? nothing : channel,
            Int(data["filter_order"]),
        )
    end
    kind == "literal" && return LiteralNode(data["value"])
    kind == "call" || throw(ArgumentError("unknown expression node kind: $kind"))

    args_data = data["args"]
    arg_keys = sort!(collect(keys(args_data)); by = key -> parse(Int, key))
    args = Tuple(_decode_expression(args_data[key]) for key in arg_keys)

    kwargs_data = data["kwargs"]
    kw_keys = sort!(collect(keys(kwargs_data)))
    kw_names = Tuple(Symbol(key) for key in kw_keys)
    kw_values = Tuple(kwargs_data[key] for key in kw_keys)
    kwargs = NamedTuple{kw_names}(kw_values)

    op = _op_from_val(Val(Symbol(data["op"])))
    return CallNode(op, args, kwargs)
end
