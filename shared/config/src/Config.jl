module Config

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
export freq_bands, depths
export use_misfit!, phase_fields, polarity_fields
export operator_module, output_field, bases_of, is_composed, channel_of
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

# Misfit operator plugin loader

const _MISFIT_DIR = joinpath(@__DIR__, "..", "..", "misfit", "src")
const _LOADED_MISFIT_MODULES = String[]
const _PHASE_TYPE = Dict{Symbol, String}()
const _OPERATOR_MODULE = Dict{Symbol, Module}()   # name -> operator module
const _OUTPUT_FIELD = Dict{Symbol, Symbol}()      # name -> output field
const _BASES = Dict{Symbol, Vector{Symbol}}()     # composed name -> bases
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
output_field(name::Symbol)::Symbol = _OUTPUT_FIELD[name]
bases_of(name::Symbol) = _BASES[name]
is_composed(name::Symbol)::Bool = name in _IS_COMPOSED
channel_of(name::Symbol)::Union{String, Nothing} = get(_CHANNEL, name, nothing)

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
