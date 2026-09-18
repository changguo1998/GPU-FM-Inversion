#
# config_sample.jl — Sample pipeline configuration
#
# Copy this file and edit the return values for your dataset.
# Each function MUST be implemented — if one is missing, you'll
# get a ConfigError at startup telling you exactly what to define.
#
# The Config module is loaded by input.jl (via `using Config`) before this file.
# When running standalone for validation, uncomment the using block.
#
# Usage:
#   julia --project=. scripts/input.jl config_sample.jl

# (Uncomment only for standalone validation)
# using Config

# Objective functions - compiled into pipeline operators by input.jl
using Misfit

p_observed = observed(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
p_synthetic = synthetic(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
s_observed = observed(S; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
s_synthetic = synthetic(S; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)

Config.@objective XcorrP = 1 - maxCC(p_observed, p_synthetic; maxlag = 3)
Config.@objective XcorrS = 1 - maxCC(s_observed, s_synthetic; maxlag = 3)
# TODO(deferred): Polarity/Psr and AbsShiftP/S/RelShift are not registered.

Config.freq_bands() = [(0.5, 2.0)]
Config.durations() = [0.1, 0.2, 0.3]  # Gaussian STF σ (s)

# Depth range
Config.depths() = [5.0, 10.0, 15.0]

# 震相字段映射 (PhasePick struct 字段名 → 震相类型)
# phase_fields: 走时字段; polarity_fields: 极性字段 (仅含极性数据的震相)
Config.phase_fields() = Dict("P" => :P_time, "S" => :S_time)
Config.polarity_fields() = Dict("P" => :P_polarity)

# ── Data interface ──
# The initial search grid is automatically provided by the Grid module.

# Event information
Config.load_event() = begin
    # Return IO.EventInfo(longitude, latitude, depth, magnitude, origintime)
    # Example: read from your own data format
    error("Implement load_event() in your config file")
end

# Station metadata
Config.load_stations() = begin
    # Return Vector{IO.StationInfo}
    error("Implement load_stations() in your config file")
end

# Phase arrival picks
Config.load_phase_picks() = begin
    # Return Vector{IO.PhasePick}
    error("Implement load_phase_picks() in your config file")
end

# Raw waveform for a given phase identifier
Config.load_waveform(phase_id::String) = begin
    # Return Vector{Float64}
    # Phase key format: {network}.{station}.{channel}.{phase_type}
    error("Implement load_waveform() in your config file")
end

# Green's functions for a source-station pair
# Return nothing if no GF available for this combination.
Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
    # Return (gf_array::Array{Float64,3}, dt::Float64, tp::Float64, ts::Float64) or nothing
    # gf_array shape: (nt, 6, 3) — channels in [N, E, D] order
    error("Implement load_gf() in your config file")
end
