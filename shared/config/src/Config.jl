module Config

# Config — Pipeline configuration interface (declarations only)
#
# Users write a config script that includes this module and implements
# each function below. Any function left unimplemented throws an error
# at runtime with a clear message describing the required return type.
#
# Usage (user's config script, e.g. my_event.jl):
#   include("shared/config/src/Config.jl")
#   using .Config
#
#   function Config.misfit_modules()      return ["XCorr", "Polarity"]      end
#   # ... etc for each function
#
# Stage scripts then include the user's config file and call interface functions.

export misfit_modules, minimum_stations, phase_type
export freq_bands, depths
export use_misfit!, phase_fields, polarity_fields
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

# Misfit module plugin loader

const _MISFIT_DIR = joinpath(@__DIR__, "..", "..", "misfit")
const _LOADED_MISFIT_MODULES = String[]
const _PHASE_TYPE = Dict{Symbol, String}()

"""
    use_misfit!(name::Symbol; from::Symbol = name)

Load a misfit module plugin and register it in `misfit_modules()`.

Loads the plugin from `shared/misfit/{from}.jl` and creates `Config.{name}`
as an inner module with config stubs and a `preprocess()` function.

When `from` differs from `name`, the plugin file is used as a template
instantiated under a new name — useful for running the same misfit
computation with different parameters (e.g. XCorr for P and S waves).

Examples:
  # Simple load
  Config.use_misfit!(:PolarityP, from = :Polarity, phase_type = "P")

  # Template instantiation — both inherit from Xcorr template
  Config.use_misfit!(:XcorrP, from = :Xcorr)
  Config.use_misfit!(:XcorrS, from = :Xcorr)
  Config.XcorrP.trim() = [-2.0, 5.0]
  Config.XcorrS.trim() = [-2.0, 8.0]

  # Explicit override of misfit_modules (optional)
  function Config.misfit_modules()
      return ["XcorrP", "XcorrS", "Polarity"]
  end
"""
function use_misfit!(
    name::Symbol;
    from::Symbol = name,
    phase_type::Union{String, Nothing} = nothing,
):Nothing
    file = joinpath(_MISFIT_DIR, "$from.jl")
    if !isfile(file)
        error("Misfit module '$name' not found at $file")
    end
    @eval module $name
    include($(file))
    end
    n = string(name)
    if !(n in _LOADED_MISFIT_MODULES)
        push!(_LOADED_MISFIT_MODULES, n)
    end
    if phase_type !== nothing
        _PHASE_TYPE[name] = phase_type
    end
    return nothing
end
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
PhasePick struct field symbol (e.g. `:P_time`, `:S_time`).

Example: `return Dict("P" => :P_time, "S" => :S_time)`
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
PhasePick polarity field symbol (e.g. `:P_polarity`).
Only phase types with polarity data need entries.

Example: `return Dict("P" => :P_polarity)`
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

Return the list of active misfit module names.

By default returns modules registered via `use_misfit!()`, in registration
order. Override this function in your config script to reorder or filter.

Example (auto-detection, no override needed):
  # Just call use_misfit! — modules are listed automatically

Example (explicit override):
  function Config.misfit_modules()
      return ["XcorrP", "Polarity"]   # exclude XcorrS
  end
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

Return list of (low_cut, high_cut) frequency-band pairs in Hz.
Example: `return [(0.5, 2.0), (1.0, 4.0)]`
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

Return list of source depths (km) for Green's function lookup.
Example: `return [5.0, 10.0, 15.0]`
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

Return Green's functions for a source-station pair.

Returns `nothing` if no GF is available for this combination (caller will
skip with a warning). On success returns a tuple:

  (gf_array, dt, tp, ts)

  gf_array  :: Array{Float64,3}   shape (nt, 6, 3) — [N, E, D] channel order
  dt        :: Float64             sampling interval (seconds)
  tp        :: Float64             P arrival time from GF start (seconds)
  ts        :: Float64             S arrival time from GF start (seconds)
"""
function load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon)::Union{Nothing, Tuple{Array{Float64,3}, Float64, Float64, Float64}}
    throw(
        ConfigError(
            "load_gf",
            "-> Union{Nothing, Tuple{Array{Float64,3}, Float64, Float64, Float64}}",
        ),
    )
end

end # module
