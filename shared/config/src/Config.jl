module Config

import Misfit

# Config — pipeline configuration interface (declarations only)
#
# Users write a config script that includes this module and implements each
# function below; unimplemented ones throw a ConfigError at runtime.
#
# Usage (user's config script, e.g. my_event.jl):
#   include("shared/config/src/Config.jl"); using .Config
#   function Config.misfit_modules()      return ["XCorr", "Polarity"]      end
#   # ... etc for each function
#
# Stage scripts include the user's config file and call interface functions.

export misfit_modules, minimum_stations, phase_type
export freq_bands, depths, durations
export use_misfit!, phase_fields, polarity_fields
export operator_module, output_field, bases_of, is_composed, channel_of, primitive_requirements
export @objective, compile_objectives!, objective, objective!, objectives
export load_event, load_stations, load_phase_picks, load_waveform, load_gf

# Error for unimplemented interface functions

struct ConfigError <: Exception
    func::String
    msg::String
end

Base.showerror(io::IO, e::ConfigError) = print(
    io,
    "ConfigError: $(e.func)() is not implemented.\n" *
    "  Your config script must define:  $(e.func)()  $(e.msg)",
)

# Target-function expression registry

const _OBJECTIVES = Dict{Symbol, Misfit.AbstractExpr}()
const _OBJECTIVE_ORDER = Symbol[]
const _COMPILED_OBJECTIVES = Set{Symbol}()
const _XCORR_BASES = Dict{Tuple, Symbol}()
const _XCORR_MAXLAG = Dict{Symbol, Float64}()

"""Register a named target-function expression."""
function objective!(name::Symbol, expr::Misfit.AbstractExpr)
    haskey(_OBJECTIVES, name) && error("objective already registered: $name")
    _OBJECTIVES[name] = expr
    push!(_OBJECTIVE_ORDER, name)
    return expr
end

"""Return a registered target-function expression."""
objective(name::Symbol)::Misfit.AbstractExpr = _OBJECTIVES[name]

"""Return a copy of all registered target-function expressions."""
objectives() = copy(_OBJECTIVES)

"""Register `@objective Name = expression` in the target-function registry."""
macro objective(definition)
    definition isa Expr && definition.head == :(=) || error("usage: @objective Name = expression")
    name, expr = definition.args
    name isa Symbol || error("objective name must be a Symbol")
    return :(objective!($(QuoteNode(name)), $(esc(expr))))
end

_is_call(expr, op, nargs) = expr isa Misfit.CallNode && expr.op isa op && length(expr.args) == nargs
_same_waveform(left, right) = all(
    getfield(left, field) == getfield(right, field) for
    field in (:phase, :band, :window, :channel, :filter_order)
)
_waveform_signature(node) = (node.phase, node.band, node.window, node.channel, node.filter_order)

function _xcorr_spec(name::Symbol, expr::Misfit.AbstractExpr)
    output = Misfit.Xcorr.CC_MAX
    call = expr
    if _is_call(expr, Misfit.SubtractOp, 2)
        expr.args[1] isa Misfit.LiteralNode && expr.args[1].value == 1 ||
            throw(ArgumentError("objective $name: XCorr misfit must be `1 - maxCC(...)`"))
        call = expr.args[2]
        _is_call(call, Misfit.MaxCCOp, 2) ||
            throw(ArgumentError("objective $name: XCorr misfit must be `1 - maxCC(...)`"))
    elseif _is_call(expr, Misfit.MaxCCOp, 2)
        call = expr
    elseif _is_call(expr, Misfit.LagCCOp, 2)
        output = Misfit.Xcorr.BEST_LAG
    else
        throw(ArgumentError("objective $name: unsupported XCorr expression"))
    end

    cc = call
    keys(cc.kwargs) == (:maxlag,) ||
        throw(ArgumentError("objective $name: maxCC requires only the `maxlag` keyword"))

    observed_node, synthetic_node = cc.args
    observed_node isa Misfit.WaveformNode && observed_node.role == :observed ||
        throw(ArgumentError("objective $name: first maxCC argument must be observed(...)"))
    synthetic_node isa Misfit.WaveformNode && synthetic_node.role == :synthetic ||
        throw(ArgumentError("objective $name: second maxCC argument must be synthetic(...)"))

    _same_waveform(observed_node, synthetic_node) ||
        throw(ArgumentError("objective $name: observed and synthetic waveform settings must match"))
    observed_node.phase in (:P, :S) ||
        throw(ArgumentError("objective $name: only P and S phases are supported"))

    maxlag = Float64(cc.kwargs.maxlag)
    maxlag > 0 || throw(ArgumentError("objective $name: maxlag must be positive"))
    return observed_node, maxlag, output
end

_is_xcorr_root(expr) =
    _is_call(expr, Misfit.MaxCCOp, 2) ||
    _is_call(expr, Misfit.LagCCOp, 2) ||
    (
        _is_call(expr, Misfit.SubtractOp, 2) &&
        expr.args[1] isa Misfit.LiteralNode &&
        expr.args[1].value == 1 &&
        _is_call(expr.args[2], Misfit.MaxCCOp, 2)
    )

function _validate_band(name, waveform)
    bands = freq_bands()
    length(bands) == 1 || throw(
        ArgumentError("objective $name: the first DSL version requires exactly one frequency band"),
    )
    Tuple(Float64.(bands[1])) == waveform.band || throw(
        ArgumentError(
            "objective $name: waveform band $(waveform.band) is not Config.freq_bands()[1]",
        ),
    )
    return nothing
end

function _compile_xcorr!(name::Symbol, expr::Misfit.AbstractExpr)
    waveform, maxlag, output = _xcorr_spec(name, expr)
    _validate_band(name, waveform)
    haskey(_OPERATOR_MODULE, name) &&
        error("objective $name conflicts with an existing misfit registration")

    use_misfit!(
        name;
        operator = Misfit.Xcorr,
        phase = string(waveform.phase),
        output = output,
        channel = waveform.channel,
    )
    instance = _instance_module(name)
    window = collect(waveform.window)
    filter_order = waveform.filter_order
    Core.eval(instance, :(trim() = $window))
    Core.eval(instance, :(max_lag_periods() = $maxlag))
    Core.eval(instance, :(filter_order() = $filter_order))
    Core.eval(instance, :(band_low() = Int32[1]))
    Core.eval(instance, :(band_high() = Int32[2]))
    get!(_XCORR_BASES, _waveform_signature(waveform), name)
    _XCORR_MAXLAG[name] = maxlag
    return nothing
end

function _base_for(name, waveform)
    base = get(_XCORR_BASES, _waveform_signature(waveform), nothing)
    base === nothing &&
        throw(ArgumentError("objective $name: register a matching XCorr waveform objective first"))
    return base
end

function _pipeline_waveform_base!(name, node, bases)
    node isa Misfit.WaveformNode ||
        throw(ArgumentError("objective $name: waveform primitive requires a waveform source"))
    node.role in (:observed, :synthetic) ||
        throw(ArgumentError("objective $name: unsupported waveform role $(node.role)"))
    base = _base_for(name, node)
    base in bases || push!(bases, base)
    return base
end

function _validate_pipeline_expression!(name, expr, bases, primitives)
    expr isa Misfit.LiteralNode && return nothing
    expr isa Misfit.InputNode &&
        throw(ArgumentError("objective $name: named input nodes are unavailable in the pipeline"))
    expr isa Misfit.WaveformNode && throw(
        ArgumentError("objective $name: waveform sources must be consumed by a waveform primitive"),
    )
    expr isa Misfit.CallNode ||
        throw(ArgumentError("objective $name: unsupported expression node $(typeof(expr))"))

    op = expr.op
    if op isa Union{Misfit.MaxCCOp, Misfit.LagCCOp}
        length(expr.args) == 2 || throw(ArgumentError("objective $name: CC requires two waveforms"))
        observed_node, synthetic_node = expr.args
        observed_node isa Misfit.WaveformNode && observed_node.role == :observed ||
            throw(ArgumentError("objective $name: first CC argument must be observed(...)"))
        synthetic_node isa Misfit.WaveformNode && synthetic_node.role == :synthetic ||
            throw(ArgumentError("objective $name: second CC argument must be synthetic(...)"))
        _same_waveform(observed_node, synthetic_node) ||
            throw(ArgumentError("objective $name: observed and synthetic CC settings must match"))
        keys(expr.kwargs) == (:maxlag,) ||
            throw(ArgumentError("objective $name: CC requires only the `maxlag` keyword"))
        base = _pipeline_waveform_base!(name, observed_node, bases)
        Float64(expr.kwargs.maxlag) == _XCORR_MAXLAG[base] ||
            throw(ArgumentError("objective $name: CC maxlag must match base objective $base"))
    elseif op isa Union{Misfit.AmpScaleOp, Misfit.SignScaleOp}
        length(expr.args) == 1 ||
            throw(ArgumentError("objective $name: waveform scale requires one argument"))
        _pipeline_waveform_base!(name, only(expr.args), bases)
    elseif op isa Union{Misfit.EnergyOp, Misfit.RMSOp}
        length(expr.args) == 1 ||
            throw(ArgumentError("objective $name: energy/RMS requires one argument"))
        arg = only(expr.args)
        if arg isa Misfit.WaveformNode
            _pipeline_waveform_base!(name, arg, bases)
        else
            _validate_pipeline_expression!(name, arg, bases, primitives)
        end
    elseif op isa Union{
        Misfit.AddOp,
        Misfit.SubtractOp,
        Misfit.MultiplyOp,
        Misfit.DivideOp,
        Misfit.PowerOp,
        Misfit.NegateOp,
        Misfit.Abs2Op,
        Misfit.AbsOp,
        Misfit.LogOp,
        Misfit.Log10Op,
        Misfit.SignOp,
    }
        for arg in expr.args
            _validate_pipeline_expression!(name, arg, bases, primitives)
        end
    else
        throw(ArgumentError("objective $name: unsupported pipeline operator $(typeof(op))"))
    end
    push!(primitives, Symbol(Misfit._op_name(op)))
    return nothing
end

function _compile_expression!(name, expr)
    bases = Symbol[]
    primitives = Set{Symbol}()
    _validate_pipeline_expression!(name, expr, bases, primitives)
    sort!(bases; by = base -> (get(_PHASE_TYPE, base, ""), string(base)))
    use_misfit!(name; operator = Misfit.Expression, bases = bases, output = Misfit.Expression.VALUE)
    _PRIMITIVES[name] = sort!(collect(primitives); by = string)
    return nothing
end

function _compile_objective!(name::Symbol, expr::Misfit.AbstractExpr)
    if _is_xcorr_root(expr)
        return _compile_xcorr!(name, expr)
    end
    return _compile_expression!(name, expr)
end

"""Compile registered objective expressions into pipeline operator instances."""
function compile_objectives!()
    for compile_bases in (true, false)
        for name in _OBJECTIVE_ORDER
            name in _COMPILED_OBJECTIVES && continue
            _is_xcorr_root(_OBJECTIVES[name]) == compile_bases || continue
            _compile_objective!(name, _OBJECTIVES[name])
            push!(_COMPILED_OBJECTIVES, name)
        end
    end
    return nothing
end

# Misfit operator plugin loader

const _MISFIT_DIR = joinpath(@__DIR__, "..", "..", "misfit", "src")
const _LOADED_MISFIT_MODULES = String[]
const _PHASE_TYPE = Dict{Symbol, String}()
const _OPERATOR_MODULE = Dict{Symbol, Module}()   # name -> operator module
const _INSTANCE_MODULE = Dict{Symbol, Module}()   # name -> generated Config submodule
const _OUTPUT_FIELD = Dict{Symbol, Symbol}()      # name -> output field
const _BASES = Dict{Symbol, Vector{Symbol}}()     # composed name -> bases
const _PRIMITIVES = Dict{Symbol, Vector{Symbol}}() # composed name -> required DSL primitives
const _IS_COMPOSED = Set{Symbol}()
const _CHANNEL = Dict{Symbol, String}()           # name -> channel filter (Level 1, optional)

# operator module -> template file path
_operator_template_path(op::Module) = joinpath(_MISFIT_DIR, "$(nameof(op)).jl")

"""
    use_misfit!(name; operator, output, phase=nothing, bases=nothing, channel=nothing)

Register a misfit instance. Level 1 (base): `operator` + `phase` + `output`.
Level 2 (composed): `operator` (aggregate) + `bases` + `output`.

`output` must be in `operator.outputs()`. Level 1 creates `Config.{name}` instance
module (per-instance parameter overrides via `Config.{name}.trim() = ...`).
Level 2 does not create an instance module.
"""
function use_misfit!(
    name::Symbol;
    operator::Module,
    output::Symbol,
    phase::Union{String, Nothing} = nothing,
    bases = nothing,
    channel::Union{String, Nothing} = nothing,
)
    avail = operator.outputs()
    output ∈ avail ||
        error("use_misfit!($(name)): output $output not in $(nameof(operator)).outputs() ($avail)")

    if bases === nothing
        # Level 1: include operator template into instance module
        tmpl = _operator_template_path(operator)
        @eval module $(name)
        include($(tmpl))
        end
        _INSTANCE_MODULE[name] = Base.invokelatest(getfield, @__MODULE__, name)
        _PHASE_TYPE[name] = phase
        channel !== nothing && (_CHANNEL[name] = channel)
    else
        push!(_IS_COMPOSED, name)
        _BASES[name] = bases
    end

    _OPERATOR_MODULE[name] = operator
    _OUTPUT_FIELD[name] = output
    n = string(name)
    !(n in _LOADED_MISFIT_MODULES) && push!(_LOADED_MISFIT_MODULES, n)
    return nothing
end

# Accessors
operator_module(name::Symbol)::Module = _OPERATOR_MODULE[name]
_instance_module(name::Symbol)::Module = _INSTANCE_MODULE[name]
output_field(name::Symbol)::Symbol = _OUTPUT_FIELD[name]
bases_of(name::Symbol) = _BASES[name]
is_composed(name::Symbol)::Bool = name in _IS_COMPOSED
channel_of(name::Symbol)::Union{String, Nothing} = get(_CHANNEL, name, nothing)
primitive_requirements(name::Symbol) = get(_PRIMITIVES, name, Symbol[])

"""
    phase_type(name::Symbol) -> Union{String, Nothing}

Return the phase type declared for a misfit module instance, or `nothing`
if none was declared.
"""
function phase_type(name::Symbol)::Union{String, Nothing}
    return get(_PHASE_TYPE, name, nothing)
end

# ===== 震相字段映射 (由用户 config.jl 定义) =====

"""
    phase_fields() -> Dict{String, Symbol}

Return a dict mapping phase type string (e.g. "P", "S") to the corresponding
PhasePick struct field symbol, e.g. `Dict("P" => :P_time, "S" => :S_time)`.
"""
function phase_fields()::Dict{String, Symbol}
    throw(
        ConfigError(
            "phase_fields",
            "-> Dict{String, Symbol}  (e.g. return Dict(\"P\" => :P_time, \"S\" => :S_time))",
        ),
    )
end

"""
    polarity_fields() -> Dict{String, Symbol}

Return a dict mapping phase type string (e.g. "P") to the corresponding
PhasePick polarity field symbol (e.g. `:P_polarity`); only phase types with
polarity data need entries. E.g. `Dict("P" => :P_polarity)`.
"""
function polarity_fields()::Dict{String, Symbol}
    throw(
        ConfigError(
            "polarity_fields",
            "-> Dict{String, Symbol}  (e.g. return Dict(\"P\" => :P_polarity))",
        ),
    )
end

# Interface functions (must be implemented by user config)

"""
    misfit_modules() -> Vector{String}

Return the list of active misfit module names: by default, modules registered
via `use_misfit!()` in registration order. Override in the config to reorder
or filter (e.g. `return ["XcorrP", "Polarity"]` to exclude XcorrS).
"""
function misfit_modules()::Vector{String}
    return copy(_LOADED_MISFIT_MODULES)
end

"""
    minimum_stations() -> Int

Minimum station count required for misfit accumulation.
"""
function minimum_stations()::Int
    throw(ConfigError("minimum_stations", "-> Int  (e.g. return 2)"))
end

"""
    freq_bands() -> Vector{Tuple{Float64, Float64}}

Return list of (low_cut, high_cut) frequency-band pairs in Hz, e.g. `[(0.5, 2.0), (1.0, 4.0)]`.
"""
function freq_bands()::Vector{Tuple{Float64, Float64}}
    throw(
        ConfigError(
            "freq_bands",
            "-> Vector{Tuple{Float64,Float64}}  (e.g. return [(0.5, 2.0), (1.0, 4.0)])",
        ),
    )
end

"""
    depths() -> Vector{Float64}

Return list of source depths (km) for Green's function lookup, e.g. `[5.0, 10.0, 15.0]`.
"""
function depths()::Vector{Float64}
    throw(ConfigError("depths", "-> Vector{Float64}  (e.g. return [5.0, 10.0, 15.0])"))
end

"""
    durations() -> Vector{Float64}

Return Gaussian STF duration candidates as σ in seconds, e.g. `[0.1, 0.2, 0.3]`.
"""
function durations()::Vector{Float64}
    throw(ConfigError("durations", "-> Vector{Float64}  (e.g. return [0.1, 0.2, 0.3])"))
end

"""
    load_event() -> IO.EventInfo

Return event information (location, magnitude, origin time).
"""
function load_event()::IO.EventInfo
    throw(
        ConfigError(
            "load_event",
            "-> IO.EventInfo  (longitude, latitude, depth, magnitude, origintime)",
        ),
    )
end

"""
    load_stations() -> Vector{IO.StationInfo}

Return station metadata for all stations/channels.
"""
function load_stations()::Vector{IO.StationInfo}
    throw(
        ConfigError(
            "load_stations",
            "-> Vector{IO.StationInfo}  (id, network, station, channel, ...)",
        ),
    )
end

"""
    load_phase_picks() -> Vector{IO.PhasePick}

Return phase arrival picks (P/S times and P polarity) for each station.
"""
function load_phase_picks()::Vector{IO.PhasePick}
    throw(
        ConfigError(
            "load_phase_picks",
            "-> Vector{IO.PhasePick}  (station_id, P_time, S_time, P_polarity)",
        ),
    )
end

"""
    load_waveform(phase_id::String) -> Vector{Float64}

Return the raw observed waveform for a given phase identifier.
Phase key format: `{network}.{station}.{channel}.{phase_type}`.
"""
function load_waveform(phase_id::String)::Vector{Float64}
    throw(ConfigError("load_waveform", "-> Vector{Float64}  (raw waveform for the given phase_id)"))
end

"""
    load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) -> Union{Nothing, Tuple{Array{Float64,3}, Float64, Float64, Float64}}

Return Green's functions for a source-station pair: `nothing` if unavailable
(caller skips with a warning), else a tuple `(gf_array, dt, tp, ts)` with
`gf_array` `[nt × 6 × 3]` [N, E, D], `dt` sampling interval (s), and `tp`/`ts`
P/S arrival times from GF start (s).
"""
function load_gf(
    src_lat,
    src_lon,
    src_depth,
    sta_lat,
    sta_lon,
)::Union{Nothing, Tuple{Array{Float64, 3}, Float64, Float64, Float64}}
    throw(
        ConfigError(
            "load_gf",
            "-> Union{Nothing, Tuple{Array{Float64,3}, Float64, Float64, Float64}}",
        ),
    )
end

end # module
