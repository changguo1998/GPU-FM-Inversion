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

export misfit_modules, minimum_stations
export freq_bands, depths
export xcorr_params, polarity_params
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

# Interface functions (must be implemented by user config)

"""
    misfit_modules() -> Vector{String}

Return the list of active misfit module names.
Example: `return ["XCorr", "Polarity"]`
"""
function misfit_modules()::Vector{String}
    throw(
        ConfigError("misfit_modules", "-> Vector{String}  (e.g. return [\"XCorr\", \"Polarity\"])"),
    )
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
    xcorr_params() -> NamedTuple{(:maxlag_factor, :filter_order,
                                  :P_trim, :S_trim,
                                  :select_threshold, :deselect_threshold)}

Return XCorr module parameters.

Fields:
  maxlag_factor     :: Float64   fraction of window for max lag
  filter_order      :: Int       Butterworth filter order
  P_trim            :: Vector{Float64}   P-wave trim window [pre, post] seconds
  S_trim            :: Vector{Float64}   S-wave trim window [pre, post] seconds
  select_threshold  :: Float64   CC threshold to select a phase
  deselect_threshold :: Float64  CC threshold to deselect a phase

Example:
  return (maxlag_factor=0.5, filter_order=4,
          P_trim=[-2.0, 5.0], S_trim=[-2.0, 5.0],
          select_threshold=0.5, deselect_threshold=0.3)
"""
function xcorr_params()
    throw(
        ConfigError(
            "xcorr_params",
            "-> NamedTuple (maxlag_factor, filter_order, P_trim, S_trim, select_threshold, deselect_threshold)",
        ),
    )
end

"""
    polarity_params() -> NamedTuple{(:trim,), <:NTuple{1}}

Return Polarity module parameters.

Fields:
  trim :: Vector{Float64}   [start, end] seconds after P arrival

Example:
  return (trim=[0.0, 2.0],)
"""
function polarity_params()
    throw(ConfigError("polarity_params", "-> NamedTuple (trim=[t_start, t_end])"))
end

"""
    load_event() -> IO.EventInfo

Return event information (location, magnitude, origin time).
"""
function load_event()
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
function load_stations()
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
function load_phase_picks()
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
function load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon)
    throw(
        ConfigError(
            "load_gf",
            "-> Union{Nothing, Tuple{Array{Float64,3}, Float64, Float64, Float64}}",
        ),
    )
end

end # module
