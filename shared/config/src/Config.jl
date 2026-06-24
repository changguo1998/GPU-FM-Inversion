module Config

# =============================================================================
# Config — Pipeline configuration interface (declarations only)
#
# Users write a config script that includes this module and implements
# each function below.  Any function left unimplemented throws an error
# at runtime with a clear message describing the required return type.
#
# Usage (user's config script, e.g. my_event.jl):
#
#   include("shared/config/src/Config.jl")
#   using .Config
#
#   function Config.misfit_modules()
#       return ["XCorr", "Polarity"]
#   end
#   # ... etc for each function
#
# The stage scripts then include the user's config file and call the
# interface functions.
# =============================================================================

export misfit_modules, module_weights, minimum_stations
export freq_bands, depths, grid_params
export xcorr_params, polarity_params, greens_params
export data_file

# ─────────────────────────────────────────────────────────
# Error for unimplemented interface functions
# ─────────────────────────────────────────────────────────

struct ConfigError <: Exception
    func::String
    msg::String
end

Base.showerror(io::IO, e::ConfigError) = print(
    io,
    "ConfigError: $(e.func)() is not implemented.\n" *
    "  Your config script must define:  $(e.func)()  $(e.msg)",
)

# ─────────────────────────────────────────────────────────
# Interface functions (must be implemented by user config)
# ─────────────────────────────────────────────────────────

"""
    misfit_modules() -> Vector{String}

Return the list of active misfit module names.
Example: `return [\"XCorr\", \"Polarity\"]`
"""
function misfit_modules()::Vector{String}
    throw(
        ConfigError("misfit_modules", "-> Vector{String}  (e.g. return [\"XCorr\", \"Polarity\"])"),
    )
end

"""
    module_weights() -> Vector{Float64}

Return weights for each misfit module, in the same order as `misfit_modules()`.
Must sum to 1.0 (validated by caller).
Example: `return [0.5, 0.25, 0.25]`
"""
function module_weights()::Vector{Float64}
    throw(ConfigError("module_weights", "-> Vector{Float64}  (e.g. return [0.5, 0.25, 0.25])"))
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
    grid_params() -> NamedTuple{(:strike0, :dstrike, :nstrike,
                                 :dip0, :ddip, :ndip,
                                 :rake0, :drake, :nrake), <:NTuple{9}}

Return initial grid-search parameters. All angles in degrees.

Fields:
  strike0  :: Float64   start of strike range [0, 360)
  dstrike  :: Float64   step size for strike
  nstrike  :: Int       number of strike grid points
  dip0     :: Float64   start of dip range [0, 90]
  ddip     :: Float64   step size for dip
  ndip     :: Int       number of dip grid points
  rake0    :: Float64   start of rake range [-90, 90]
  drake    :: Float64   step size for rake
  nrake    :: Int       number of rake grid points

Example:
  return (strike0=45.0, dstrike=20.0, nstrike=3,
          dip0=30.0, ddip=20.0, ndip=3,
          rake0=0.0, drake=20.0, nrake=3)
"""
function grid_params()
    throw(
        ConfigError(
            "grid_params",
            "-> NamedTuple (strike0, dstrike, nstrike, dip0, ddip, ndip, rake0, drake, nrake)",
        ),
    )
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
    greens_params() -> NamedTuple{(:gf_dir, :model), <:NTuple{2}}

Return Green's function file parameters.

Fields:
  gf_dir :: String   directory containing per-phase GF HDF5 files
  model  :: String   velocity model identifier

Example:
  return (gf_dir=\"data/greens/\", model=\"iasp91\")
"""
function greens_params()
    throw(ConfigError("greens_params", "-> NamedTuple (gf_dir=\"path/\", model=\"name\")"))
end

"""
    data_file() -> String

Path to the external HDF5 data file containing event info, station metadata,
phase picks, and raw waveforms. Same schema as the former `raw.h5`.

Example: `return "data/my_event.h5"`
"""
function data_file()::String
    throw(ConfigError("data_file", "-> String  (e.g. return \"data/my_event.h5\")"))
end

end # module
