#!/usr/bin/env julia
#
# input.jl - Data ingestion and initialization stage.
# Loads event/station/waveform/Green's-function data from the user config,
# preprocesses waveforms, and writes database.h5 plus the initial strategy.
#
# Usage:
#   julia scripts/input.jl <config.jl>

# === 1. Import dependencies ===
using HDF5
using LinearAlgebra
using Dates

using StageLog

using IO, Signal, Config, Search, Misfit

# === 2. Parse arguments and configure logging ===
config_jl = ARGS[1]
data_dir = dirname(abspath(config_jl))

StageLog.setup_logger!("input", joinpath(data_dir, "input.log"))

@info "="^70
@info "input stage started"
@info "  config   = $config_jl"
@info "  data dir = $data_dir"

# === 3. Load user configuration (config.jl) ===
# The config registers objectives and defines the external data interface.
include(abspath(config_jl))
Config.compile_objectives!()

# Read pipeline parameters from the configuration.
misfit_modules = Config.misfit_modules()
freq_bands = Config.freq_bands()
depths = Config.depths()
durations = Config.durations()

n_bands = length(freq_bands)
n_depths = length(depths)
n_durations = length(durations)
isempty(durations) && error("Config.durations() must contain at least one STF duration")
all(durations .>= 0.0) || error("Config.durations() values must be non-negative")

# Build the frequency axis from unique band boundaries.
freq_vals = sort(unique(Float64[v for (low, high) in freq_bands for v in (low, high)]))

@info "Config loaded"
@info "  misfit_modules = $misfit_modules"
@info "  freq_bands     = $freq_bands"
@info "  depths         = $depths"
@info "  durations      = $durations s (Gaussian σ)"

# === 4. Read external data (via Config.load_*()) ===
@info "Reading external data via Config.load_*() ..."

event = Config.load_event()          # Event location, depth, magnitude, and origin time.
picks = Config.load_phase_picks()    # P/S arrivals and P-wave polarity.
stations = Config.load_stations()    # Station metadata.

station_to_idx = Dict(pick.station_id => i for (i, pick) in enumerate(picks))

n_stations = length(stations)
n_picks = length(picks)

# === 5. Build the phase list ===
# Phase-field mappings are defined by config.jl.
pf = Config.phase_fields()

ch_map = Dict("N" => 1, "E" => 2, "Z" => 3)  # GF channel order [N, E, D].


phase_list = Tuple{String, String, Int}[]
for (si, s) in enumerate(stations)
    pi = get(station_to_idx, s.id, 0)
    pi == 0 && continue
    pick = picks[pi]
    ch_name = s.channel
    for (ptype, field) in pf
        time_str = getfield(pick, field)
        isempty(time_str) && continue
        pid = "$(s.network).$(s.station).$(ch_name).$(ptype)"
        push!(phase_list, (pid, ptype, si))
    end
end

n_phases = length(phase_list)

# Extract the unique phase types present in the picks.
phase_types = sort(unique([pt for (_, pt, _) in phase_list]))


@info "  event    = (lon=$(event.longitude), lat=$(event.latitude), depth=$(event.depth), M=$(event.magnitude))"
@info "  stations = $n_stations"
@info "  phases   = $n_phases"

# === 6. Build /station (deduplicated physical stations) ===

# Deduplicate by station.id while preserving input order.
seen_ids = Set{String}()
phys_stations = IO.StationInfo[]
phys_id_to_idx = Dict{String, Int}()
for s in stations
    if s.id in seen_ids
        continue
    end
    push!(seen_ids, s.id)
    phys_id_to_idx[s.id] = length(phys_stations) + 1
    push!(phys_stations, s)
end
n_phys_stations = length(phys_stations)

# Map every input station row to its physical-station index.
full_to_phys = Int32[phys_id_to_idx[s.id] for s in stations]

# /station table (N_phys_stations rows).
phys_station_dict = Dict{String, Vector}()
phys_station_dict["id"] = [s.id for s in phys_stations]
phys_station_dict["network"] = [s.network for s in phys_stations]
phys_station_dict["station"] = [s.station for s in phys_stations]
phys_station_dict["latitude"] = [s.latitude for s in phys_stations]
phys_station_dict["longitude"] = [s.longitude for s in phys_stations]
phys_station_dict["elevation"] = [s.elevation for s in phys_stations]
phys_station_dict["dt"] = [s.dt for s in phys_stations]
phys_station_dict["begin_time"] = [s.begin_time for s in phys_stations]
phys_station_dict["distance"] = [
    IO.haversine_distance(event.latitude, event.longitude, s.latitude, s.longitude) for
    s in phys_stations
]
phys_station_dict["azimuth"] = [
    IO.compute_azimuth(event.latitude, event.longitude, s.latitude, s.longitude) for
    s in phys_stations
]
_empty_pick = IO.PhasePick("", "", "", Int8(-128))
phys_station_dict["P_time"] =
    [get(picks, get(station_to_idx, s.id, 0), _empty_pick).P_time for s in phys_stations]
phys_station_dict["S_time"] =
    [get(picks, get(station_to_idx, s.id, 0), _empty_pick).S_time for s in phys_stations]
phys_station_dict["P_polarity"] =
    [get(picks, get(station_to_idx, s.id, 0), _empty_pick).P_polarity for s in phys_stations]

@info "  station dict built ($n_stations channels, $n_phys_stations physical stations)"

# === 7. Load raw waveforms -> /channel ===
# Raw waveforms are persisted and then used as preprocessing inputs.
@info "Building /channel - raw waveforms ..."

channel_data = Dict{String, Vector{Float64}}()
seen_ch = Set{String}()
for (pid, ptype, si) in phase_list
    s = stations[si]
    ch_id = "$(s.network).$(s.station).$(s.channel)"
    if ch_id in seen_ch
        continue
    end
    push!(seen_ch, ch_id)
    wf = Config.load_waveform(pid)
    channel_data[ch_id] = wf
end

@info "  $(length(channel_data)) channels loaded"

# === 8. Load Green's functions -> /gf/{depth}/{channel_id} ===
# Config.load_gf() returns the requested component from the third dimension.
@info "Loading Green's functions via Config.load_gf() ..."

"""Load Green's functions for every station/depth pair and select each channel."""
function load_greens_functions(stations, ch_map, event, depths)
    gf_data = Dict{Float64, Dict{String, Matrix{Float64}}}()
    n_skip = 0
    n_load = 0
    for s in stations
        ch_id = "$(s.network).$(s.station).$(s.channel)"
        ch_idx = get(ch_map, s.channel, 3)
        for depth_val in depths
            result =
                Config.load_gf(event.latitude, event.longitude, depth_val, s.latitude, s.longitude)
            if result === nothing
                @warn "No GF for $ch_id at depth $depth_val km, skipping"
                n_skip += 1
                continue
            end
            gf_array, dt_gf, tp_gf, ts_gf = result
            if !haskey(gf_data, depth_val)
                gf_data[depth_val] = Dict{String, Matrix{Float64}}()
            end
            gf_data[depth_val][ch_id] = gf_array[:, :, ch_idx]
            n_load += 1
        end
    end
    @info "  GF loaded: $n_load (ch,depth) pairs, $n_skip skipped"
    return gf_data
end

gf_data = load_greens_functions(stations, ch_map, event, depths)

# === 9. Preprocess waveforms ===
# Apply bandpass filtering, trimming, and GF preprocessing per module/band/phase.
@info "Preprocessing waveforms ..."

# Build the unified module registry.
module_instances = Dict{String, Module}()
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) && continue   # Level 2 objectives have no instance module.
    module_instances[m_name] = Config._instance_module(sym)
end

# === 9a. Layer 0: shared full-waveform preprocessing ===
# Full-waveform demean/detrend/taper/bandpass per band, independent of operators.
@info "Layer 0: shared preprocessing (demean/detrend/taper/bandpass) per band ..."
ch_dt = Dict{String, Float64}()
for s in stations
    ch_dt["$(s.network).$(s.station).$(s.channel)"] = s.dt
end

# collect unique (low_cut, high_cut) bands from all freq-dependent modules
all_bands = Set{Tuple{Float64, Float64}}()
for (_, mod) in module_instances
    mod.is_freq_dependent() || continue
    bl, bh = mod.band_low(), mod.band_high()
    for i in 1:length(bl)
        push!(all_bands, (freq_vals[bl[i]], freq_vals[bh[i]]))
    end
end

prepro_obs = Dict{Tuple{Float64, Float64}, Dict{String, Vector{Float64}}}()       # (lo,hi) -> ch_id -> wf
prepro_gf = Dict{Tuple{Float64, Float64, Int}, Dict{Float64, Dict{String, Matrix{Float64}}}}()  # (lo,hi,duration_idx) -> depth -> ch_id -> gf
for (lo, hi) in all_bands
    po = Dict{String, Vector{Float64}}()
    for (ch_id, wf_raw) in channel_data
        po[ch_id] = Signal.preprocess_waveform!(copy(wf_raw), ch_dt[ch_id], lo, hi)
    end
    prepro_obs[(lo, hi)] = po
    for (duration_idx, duration) in enumerate(durations)
        pg = Dict{Float64, Dict{String, Matrix{Float64}}}()
        for d in depths
            pd = Dict{String, Matrix{Float64}}()
            for (ch_id, gf_raw) in gf_data[d]
                g = copy(gf_raw)
                for c in 1:size(g, 2)
                    g[:, c] = Signal.convolve_gaussian_stf(g[:, c], duration, ch_dt[ch_id])
                    g[:, c] = Signal.preprocess_waveform!(g[:, c], ch_dt[ch_id], lo, hi)
                end
                pd[ch_id] = g
            end
            pg[d] = pd
        end
        prepro_gf[(lo, hi, duration_idx)] = pg
    end
end

# Polarity and PSR expressions reuse XCorr's preprocessed windows and reductions.
@info "  Layer 0 complete: $(length(prepro_obs)) bands x $n_durations durations x $(length(channel_data)) channels"

# Shared preprocessing context (NamedTuple keeps helper signatures compact).
ctx = (
    stations = stations,
    picks = picks,
    station_to_idx = station_to_idx,
    depths = depths,
    freq_vals = freq_vals,
    pf = pf,
    prepro_obs = prepro_obs,
    prepro_gf = prepro_gf,
    durations = durations,
)

"""Merge one frequency/duration result into the accumulated module result."""
function merge_duration_result(prev, r, duration_idx)
    if prev === nothing
        prev = Dict{String, Any}(
            "channel_id" => r["channel_id"],
            "station_idx" => r["station_idx"],
            "obs" => Dict{Int, Any}(),
            "gf" => Dict{Float64, Any}(),
            "synamp_lag" => Dict{Float64, Any}(),
            "dot_obs_gf_lag" => Dict{Int, Any}(),
        )
    else
        @assert prev["channel_id"] == r["channel_id"]
        @assert prev["station_idx"] == r["station_idx"]
    end
    for (band_key, band_data) in r["obs"]
        get!(prev["obs"], band_key, band_data)
    end
    for (d, bands) in r["gf"]
        depth_out = get!(prev["gf"], d, Dict{Int, Any}())
        for (band_key, band_data) in bands
            band_out = get!(depth_out, band_key, Dict{Int, Any}())
            band_out[duration_idx] = band_data
        end
    end
    for (d, bands) in r["synamp_lag"]
        depth_out = get!(prev["synamp_lag"], d, Dict{Int, Any}())
        for (band_key, band_data) in bands
            band_out = get!(depth_out, band_key, Dict{Int, Any}())
            band_out[duration_idx] = band_data
        end
    end
    for (band_key, band_data) in r["dot_obs_gf_lag"]
        band_out = get!(prev["dot_obs_gf_lag"], band_key, Dict{Int, Any}())
        band_out[duration_idx] = band_data
    end
    return prev
end

"""Preprocess one module and merge its result; return nothing when no data is valid.

Frequency-dependent modules are merged across bands; other modules run once."""
function preprocess_module(mod, phases_pt, ptype, ctx, prev)
    if mod.is_freq_dependent()
        band_low = mod.band_low()
        band_high = mod.band_high()
        result = prev
        for local_idx in 1:length(band_low)
            lo = ctx.freq_vals[band_low[local_idx]]
            hi = ctx.freq_vals[band_high[local_idx]]
            for duration_idx in eachindex(ctx.durations)
                r = mod.process(
                    phases_pt,
                    ptype,
                    ctx.stations,
                    ctx.picks,
                    ctx.station_to_idx,
                    ctx.prepro_obs[(lo, hi)],
                    ctx.prepro_gf[(lo, hi, duration_idx)],
                    ctx.depths,
                    hi,
                    local_idx,
                    ctx.pf,
                )
                isempty(r["channel_id"]) && continue
                result = merge_duration_result(result, r, duration_idx)
            end
        end
        return result
    else
        error("standalone non-frequency-dependent preprocessing is not supported: $(nameof(mod))")
    end
end

# Hold preprocessing results before assembling ModuleData.
module_data = Dict{String, IO.ModuleData}()
module_results = Dict{String, Dict}()

for ptype in phase_types
    phases_pt = [(pid, si) for (pid, pt, si) in phase_list if pt == ptype]
    isempty(phases_pt) && continue
    for (m_name, mod) in module_instances
        Config.phase_type(Symbol(m_name)) != ptype && continue
        prev = get(module_results, m_name, nothing)
        result = preprocess_module(mod, phases_pt, ptype, ctx, prev)
        result !== nothing && (module_results[m_name] = result)
    end
end

@info "  preprocessing complete ($(length(misfit_modules)) modules, $(length(freq_vals)) discrete frequencies)"

# === 10. Build /event, /paraspace, and /config ===
# /paraspace stores physical values; /config stores metadata and module references.
event_dict = Dict{String, Any}(
    "longitude" => event.longitude,
    "latitude" => event.latitude,
    "depth" => event.depth,
    "magnitude" => event.magnitude,
    "origintime" => event.origintime,
)

g0 = Search.default_grid()
# Strike samples cover the full circle at 5° (0:5:355); dip/rake cover closed intervals.
strike_vals = collect(g0.strike0:g0.dstrike:(360.0 - g0.dstrike))
@assert length(strike_vals) == Int(g0.nstrike) "paraspace strike and strategy.nstrike differ"
dip_vals = Search.expand_axis(g0.dip0, g0.ddip, g0.ndip)
rake_vals = Search.expand_axis(g0.rake0, g0.drake, g0.nrake)

paraspace = Dict{String, Any}(
    "strike" => strike_vals,
    "dip" => dip_vals,
    "rake" => rake_vals,
    "depth" => Float64.(depths),
    "frequency" => freq_vals,
    "duration" => Float64.(durations),
)

db_config = Dict{String, Any}("misfit_modules" => misfit_modules)
objective_exprs = Config.objectives()
if !isempty(objective_exprs)
    db_config["objectives"] = Dict{String, Any}(
        string(name) => Misfit.encode_expression(expr) for (name, expr) in objective_exprs
    )
end

# Write /config/{ModuleName}/.
# Level 1 instances have an inner module and preprocessing parameters.
for (m_name, mod) in module_instances
    sym = Symbol(m_name)
    cfg_entry = Dict{String, Any}()
    if isdefined(mod, :trim)
        cfg_entry["trim"] = Float64.(mod.trim())
    end
    if isdefined(mod, :max_lag_periods)
        cfg_entry["max_lag_periods"] = Float64(mod.max_lag_periods())
        cfg_entry["filter_order"] = Int32(mod.filter_order())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    if isdefined(mod, :source_duration)
        cfg_entry["source_duration"] = Float64(mod.source_duration())
    end
    if isdefined(mod, :pre_P)
        cfg_entry["pre_P"] = Float64(mod.pre_P())
        cfg_entry["post_P"] = Float64(mod.post_P())
        cfg_entry["pre_S"] = Float64(mod.pre_S())
        cfg_entry["post_S"] = Float64(mod.post_S())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    cfg_entry["operator"] = string(nameof(Config.operator_module(sym)))
    cfg_entry["output"] = string(Config.output_field(sym))
    cfg_entry["is_composed"] = Int8(0)
    cfg_entry["phase"] = Config.phase_type(sym)
    ch = Config.channel_of(sym)
    cfg_entry["channel"] = ch === nothing ? "" : ch
    db_config[m_name] = cfg_entry
end

# Level 2 composed instances have no inner module or preprocessing parameters.
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) || continue
    cfg_entry = Dict{String, Any}(
        "operator" => string(nameof(Config.operator_module(sym))),
        "output" => string(Config.output_field(sym)),
        "is_composed" => Int8(1),
        "bases" => String.(Config.bases_of(sym)),
    )
    primitives = Config.primitive_requirements(sym)
    isempty(primitives) || (cfg_entry["primitives"] = String.(primitives))
    db_config[m_name] = cfg_entry
end

# Convert the result dictionary to IO.ModuleData.
function result_to_moduledata(result::Dict)::IO.ModuleData
    obs_str = Dict{String, Matrix{Float64}}()
    obs_n2_str = Dict{String, Vector{Float64}}()
    for (bk, bv) in result["obs"]
        k = string(bk)
        if haskey(bv, "obs")
            obs_val = bv["obs"]
            obs_str[k] = obs_val isa Vector ? reshape(obs_val, length(obs_val), 1) : obs_val
            if haskey(bv, "obs_norm2")
                obs_n2_str[k] = bv["obs_norm2"]
            end

        end
    end

    gf_str = Dict{Float64, Dict{String, Dict{String, Array{Float64, 3}}}}()
    syn_str = Dict{Float64, Dict{String, Dict{String, Array{Float64, 3}}}}()
    if haskey(result, "gf")
        for (d, bands) in result["gf"]
            gf_str[d] = Dict{String, Dict{String, Array{Float64, 3}}}()
            syn_str[d] = Dict{String, Dict{String, Array{Float64, 3}}}()
            for (bk, duration_data) in bands
                k = string(bk)
                gf_str[d][k] = Dict{String, Array{Float64, 3}}()
                syn_str[d][k] = Dict{String, Array{Float64, 3}}()
                for (duration_idx, bv) in duration_data
                    duration_key = string(duration_idx)
                    gf_str[d][k][duration_key] = bv["gf"]
                    if haskey(bv, "synamp")
                        syn_str[d][k][duration_key] = bv["synamp"]
                    end
                end
            end
        end
    end

    # Xcorr per-lag reductions
    synamp_lag_str = Dict{Float64, Dict{String, Dict{String, Array{Float64, 4}}}}()
    if haskey(result, "synamp_lag")
        for (d, bands) in result["synamp_lag"]
            synamp_lag_str[d] = Dict{String, Dict{String, Array{Float64, 4}}}()
            for (bk, duration_data) in bands
                band_key = string(bk)
                synamp_lag_str[d][band_key] = Dict{String, Array{Float64, 4}}()
                for (duration_idx, bv) in duration_data
                    synamp_lag_str[d][band_key][string(duration_idx)] = bv
                end
            end
        end
    end
    dog_lag_str = Dict{String, Dict{String, Array{Float64, 3}}}()
    if haskey(result, "dot_obs_gf_lag")
        for (bk, duration_data) in result["dot_obs_gf_lag"]
            band_key = string(bk)
            dog_lag_str[band_key] = Dict{String, Array{Float64, 3}}()
            for (duration_idx, bv) in duration_data
                dog_lag_str[band_key][string(duration_idx)] = bv
            end
        end
    end


    return IO.ModuleData(
        obs = obs_str,
        obs_norm2 = obs_n2_str,
        gf = gf_str,
        synamp = syn_str,
        synamp_lag = synamp_lag_str,
        dot_obs_gf_lag = dog_lag_str,
        # Legacy standalone PSR fields remain empty; active PSR is an expression over XCorr data.
        amp_P = Dict{Float64, Dict{String, Array{Float64, 3}}}(),
        amp_S = Dict{Float64, Dict{String, Array{Float64, 3}}}(),
        obs_psr = Dict{String, Vector{Float64}}(),
        channel_id = result["channel_id"],
        station_idx = full_to_phys[result["station_idx"]],
    )
end
# === 11. Write database.h5 ===
@info "Writing database.h5 ..."
db_path = joinpath(data_dir, "database.h5")
# Assemble ModuleData -> /{ModuleName}/obs + /gf.
for (m_name, result) in module_results
    module_data[m_name] = result_to_moduledata(result)
end

IO.write_database(
    db_path,
    db_config,
    event_dict,
    phys_station_dict,
    channel_data,
    gf_data,
    module_data;
    paraspace = paraspace,
)

mod_summary = join(
    [
        "$(length(module_results[mn]["channel_id"])) $mn" for
        mn in sort(collect(keys(module_results)))
    ],
    ", ",
)
@info "  phase metadata written ($mod_summary)"

# === 12. Write status_0.h5 (initial strategy) ===
# /strategy: full-space 5° SDR grid plus all depth/frequency/duration indices, iteration 0.
@info "Writing status_0.h5 ..."

strategy = IO.Strategy(
    g0.strike0,
    g0.dstrike,
    g0.nstrike,
    g0.dip0,
    g0.ddip,
    g0.ndip,
    g0.rake0,
    g0.drake,
    g0.nrake,
    Int32.(1:n_depths),
    Int32.(1:n_bands),
    Int32.(1:n_durations),
    Int32(0),
)

status0_path = joinpath(data_dir, "status_0.h5")
h5open(status0_path, "w") do f
end
IO.write_strategy(status0_path, strategy)
@info "  $status0_path written"

# === 13. Print summary ===
@info ""
@info "Stage complete:"
@info "  $(basename(db_path)) : /station ($n_phys_stations rows)"
@info "  $(basename(db_path)) : /channel ($(length(channel_data)) channels)"
@info "  $(basename(db_path)) : /gf ($(length(gf_data)) depths)"
mod_names_str = join(sort(collect(keys(module_data))), ", ")
@info "  $(basename(db_path)) : /$mod_names_str (modules)"
@info "  $(basename(db_path)) : /config, /event"
@info "  $(basename(status0_path)) : /strategy (initial grid, no trials)"
@info "  Phys stations: $n_phys_stations | Channels: $n_stations | Phases: $n_phases | Depths: $n_depths | Freq vals: $(length(freq_vals)) | Durations: $n_durations"
@info ""
@info "="^70
