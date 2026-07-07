#!/usr/bin/env julia
#
# input.jl — Data ingestion and initialization stage
#
# Runs once before the main loop:
#   1. Load config.jl (user-provided config + data interface)
#   2. Read raw data via Config.load_*() interface
#   3. Build /station, /channel, /gf, /xcorr, /polarity
#   4. Write database.h5
#   5. Write initial strategy -> status_0.h5 (NO trials)
#
# Usage:
#   julia scripts/input.jl <config.jl>

using HDF5
using LinearAlgebra
using Dates
using Random

# Logging

using StageLog

# Load shared modules

using IO, Signal, Config, Grid

# CLI

config_jl = ARGS[1]

data_dir = dirname(abspath(config_jl))

StageLog.setup_logger!("input", joinpath(data_dir, "input.log"))

@info "="^70
@info "input stage started"
@info "  config   = $config_jl"
@info "  data dir = $data_dir"

include(abspath(config_jl))

misfit_modules = Config.misfit_modules()
minimum_stations = Config.minimum_stations()
freq_bands = Config.freq_bands()
depths = Config.depths()
grid = Grid.default_grid()
xcorr = Config.xcorr_params()
polarity = Config.polarity_params()

n_bands = length(freq_bands)
n_depths = length(depths)
n_misfit_modules = length(misfit_modules)

@info "Config loaded"
@info "  misfit_modules = $misfit_modules"
@info "  freq_bands     = $freq_bands"
@info "  depths         = $depths"

# 2. Read external data

@info "Reading external data via Config.load_*() ..."

event = Config.load_event()
picks = Config.load_phase_picks()
stations = Config.load_stations()

station_to_idx = Dict(pick.station_id => i for (i, pick) in enumerate(picks))

n_stations = length(stations)
n_picks = length(picks)

# Build phase list: for each station, create P and S phase entries
# phase_list entries: (phase_id, phasetype, station_idx)

ch_map = Dict("N" => 1, "E" => 2, "Z" => 3)  # GF channel order [N, E, D]

function _channel_from_pid(pid::String)::String
    parts = split(pid, ".")
    return length(parts) >= 3 ? parts[3] : "Z"
end

phase_list = Tuple{String, String, Int}[]
for (si, s) in enumerate(stations)
    pick = picks[get(station_to_idx, s.id, 1)]
    ch_name = s.channel
    if !isempty(pick.P_time)
        pid = "$(s.network).$(s.station).$(ch_name).P"
        push!(phase_list, (pid, "P", si))
    end
    if !isempty(pick.S_time)
        pid = "$(s.network).$(s.station).$(ch_name).S"
        push!(phase_list, (pid, "S", si))
    end
end

n_phases = length(phase_list)

@info "  event    = (lon=$(event.longitude), lat=$(event.latitude), depth=$(event.depth), M=$(event.magnitude))"
@info "  stations = $n_stations"
@info "  phases   = $n_phases"

# 3. Build /station flat arrays

station_dict = Dict{String, Vector}()
station_dict["id"] = [s.id for s in stations]
station_dict["network"] = [s.network for s in stations]
station_dict["station"] = [s.station for s in stations]
station_dict["channel"] = [s.channel for s in stations]
station_dict["latitude"] = [s.latitude for s in stations]
station_dict["longitude"] = [s.longitude for s in stations]
station_dict["elevation"] = [s.elevation for s in stations]
station_dict["dt"] = [s.dt for s in stations]
station_dict["begin_time"] = [s.begin_time for s in stations]
station_dict["distance"] = [
    IO.haversine_distance(event.latitude, event.longitude, s.latitude, s.longitude) for
    s in stations
]
station_dict["azimuth"] =
    [IO.compute_azimuth(event.latitude, event.longitude, s.latitude, s.longitude) for s in stations]
station_dict["P_time"] = [pick.P_time for pick in picks]
station_dict["S_time"] = [pick.S_time for pick in picks]
station_dict["P_polarity"] = [pick.P_polarity for pick in picks]

@info "  station dict built ($n_stations stations)"

# 4. Build /channel — raw waveforms per channel

@info "Building /channel — raw waveforms ..."

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

# 5. Load Green's functions — /gf/{depth}/{channel_id}

@info "Loading Green's functions via Config.load_gf() ..."

gf_data = Dict{Float64, Dict{String, Matrix{Float64}}}()
let n_skip = 0, n_load = 0
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
end

# 6. Preprocess waveforms — build /xcorr/obs, /xcorr/gf, /polarity/obs, /polarity/gf

@info "Preprocessing waveforms ..."

P_trim = xcorr.P_trim
S_trim = xcorr.S_trim
filter_order = xcorr.filter_order
polarity_trim = polarity.trim
t_source = polarity_trim[2]

xcorr_obs = Dict{String, IO.XCorrObs}()
xcorr_gf = Dict{Float64, Dict{String, IO.XCorrGF}}()
polarity_obs = Float64[]
polarity_gf = Dict{Float64, Array{Float64, 3}}()

for freq_idx in 1:n_bands
    bnd = freq_bands[freq_idx]
    low_cut = Float64(bnd[1])
    high_cut = Float64(bnd[2])

    @info "  band $freq_idx/$n_bands: [$low_cut, $high_cut] Hz"

    for ptype in ("P", "S")
        key = "$ptype-$freq_idx"
        phases_pt = [(pid, si) for (pid, pt, si) in phase_list if pt == ptype]
        np = length(phases_pt)
        if np == 0
            continue
        end

        obs_list = Vector{Vector{Float64}}()
        obs_norm2_list = Float64[]
        gf_list = Vector{Matrix{Float64}}()

        for (pid, si) in phases_pt
            s = stations[si]
            dt = s.dt
            pick = picks[get(station_to_idx, s.id, 1)]
            ch_id = "$(s.network).$(s.station).$(s.channel)"
            wf = channel_data[ch_id]
            n_samples = length(wf)

            begin_unix = IO.parse_time_iso(s.begin_time)
            pick_time = if ptype == "P"
                IO.parse_time_iso(pick.P_time)
            else
                IO.parse_time_iso(pick.S_time)
            end
            trim_cfg = (ptype == "P") ? P_trim : S_trim

            if isnan(begin_unix) || isnan(pick_time)
                arrival_sample = n_samples ÷ 2
            else
                arrival_sample = clamp(round(Int, (pick_time - begin_unix) / dt) + 1, 1, n_samples)
            end

            pre_sec = abs(trim_cfg[1])
            post_sec = abs(trim_cfg[2])
            window_factor = max(pre_sec, post_sec) * high_cut

            # Load GF from first depth (shared by XCorr and Polarity preprocessing)
            # NOTE: known limitation — uses first-depth GF for all depths; frequency-dependent
            # filtering and polarity window should ideally differ per depth combo
            gf_full = get(gf_data[depths[1]], ch_id, nothing)
            if gf_full === nothing
                @warn "No GF for $ch_id at depth $(depths[1]), skipping preprocessing for $pid"
                continue
            end

            # XCorr preprocessing
            if "XCorr" in misfit_modules
                obs_proc, gf_proc, synamp_mat, obs_n2 = Signal.preprocess_xcorr!(
                    wf,
                    gf_full,
                    dt,
                    arrival_sample,
                    low_cut,
                    high_cut,
                    window_factor;
                    filter_order = filter_order,
                )
                push!(obs_list, obs_proc)
                push!(obs_norm2_list, obs_n2)
                push!(gf_list, gf_proc)
            end

            # Polarity preprocessing (P-wave only, first band only — polarity is frequency-independent)
            if "Polarity" in misfit_modules && ptype == "P" && freq_idx == 1
                obs_pol_val = Float64(pick.P_polarity)
                if obs_pol_val == -128.0
                    obs_pol_val = NaN
                end
                push!(polarity_obs, obs_pol_val)
                gf_pol, _ = Signal.preprocess_polarity!(
                    gf_full,
                    dt,
                    arrival_sample,
                    t_source,
                    pick.P_polarity,
                )
                # Collect gf_pol per depth (reuse first-depth GF for all depths — see note above)
                for depth_val in depths
                    if !haskey(polarity_gf, depth_val)
                        polarity_gf[depth_val] = zeros(Float64, 0, 6, 0)
                    end
                    # Append channel's gf_pol along first dim
                    if size(polarity_gf[depth_val], 1) == 0
                        polarity_gf[depth_val] = reshape(gf_pol, 1, 6, size(gf_pol, 1))
                    else
                        new_shape = (size(polarity_gf[depth_val], 1) + 1, 6, size(gf_pol, 1))
                        new_arr = zeros(Float64, new_shape)
                        new_arr[1:(end - 1), :, :] = polarity_gf[depth_val]
                        new_arr[end, :, :] = gf_pol'
                        polarity_gf[depth_val] = new_arr
                    end
                end
            end
        end

        # Stack per-phase vectors into 2D/3D arrays for /xcorr
        if "XCorr" in misfit_modules && !isempty(obs_list)
            # Truncate all to minimum length (edge phases may be shorter)
            nt_xc = minimum(length.(obs_list))
            for i in 1:length(obs_list)
                obs_list[i] = obs_list[i][1:nt_xc]
                gf_list[i] = gf_list[i][1:nt_xc, :]
            end
            obs_mat = zeros(Float64, np, nt_xc)
            gf_arr = zeros(Float64, np, 6, nt_xc)
            synamp_arr = zeros(Float64, np, 6, 6)
            for i in 1:np
                obs_mat[i, :] = obs_list[i]
                gf_arr[i, :, :] = gf_list[i]'
                synamp_arr[i, :, :] = gf_list[i]' * gf_list[i]
            end
            xcorr_obs[key] = IO.XCorrObs(obs_mat, obs_norm2_list)
            # /xcorr/gf per depth (reuse first-depth GF for all depths — 
            # known limitation: frequency-dependent filtering should differ per depth combo,
            # but input.jl uses the same filtered GF for all depths)
            for depth_val in depths
                if !haskey(xcorr_gf, depth_val)
                    xcorr_gf[depth_val] = Dict{String, IO.XCorrGF}()
                end
                xcorr_gf[depth_val][key] = IO.XCorrGF(gf_arr, synamp_arr)
            end
        end
    end
end

@info "  preprocessing complete ($n_bands bands, $(length(misfit_modules)) modules)"

# 7. Build event dict and db_config

event_dict = Dict{String, Any}(
    "longitude" => event.longitude,
    "latitude" => event.latitude,
    "depth" => event.depth,
    "magnitude" => event.magnitude,
    "origintime" => event.origintime,
)

db_config = Dict{String, Any}(
    "misfit_modules" => misfit_modules,
    "depth_vals" => Float64.(depths),
    "n_bands" => Int32(n_bands),
    "freq_bands_low" => Float64[low for (low, _) in freq_bands],
    "freq_bands_high" => Float64[high for (_, high) in freq_bands],
    "minimum_stations" => Int32(minimum_stations),
)

if "XCorr" in misfit_modules
    db_config["xcorr"] = Dict{String, Any}(
        "maxlag_factor" => Float64(xcorr.maxlag_factor),
        "filter_order" => Int32(filter_order),
        "P_trim" => Float64.(P_trim),
        "S_trim" => Float64.(S_trim),
        "select_threshold" => Float64(xcorr.select_threshold),
        "deselect_threshold" => Float64(xcorr.deselect_threshold),
    )
end

if "Polarity" in misfit_modules
    db_config["polarity"] = Dict{String, Any}("trim" => Float64.(polarity_trim))
end
# 7b. Build /index
n_phases = length(phase_list)
index_phase_ids = [p[1] for p in phase_list]
index_phase_type = [p[2] for p in phase_list]
index_station_idx = Int32[p[3] for p in phase_list]
index_distance = [station_dict["distance"][p[3]] for p in phase_list]
index_azimuth = [station_dict["azimuth"][p[3]] for p in phase_list]
# greens_depth_idx: all depths valid for all phases -> row = phase, col = depth
index_greens_depth_idx = zeros(Int32, n_phases, n_depths)
for d in 1:n_depths
    index_greens_depth_idx[:, d] .= Int32(d)
end
@info "  index built ($n_phases phases, $n_depths depths)"


# 8. Write database.h5

@info "Writing database.h5 ..."
db_path = joinpath(data_dir, "database.h5")
IO.write_database(
    db_path,
    db_config,
    event_dict,
    station_dict,
    channel_data,
    gf_data,
    xcorr_obs,
    xcorr_gf,
    polarity_obs,
    polarity_gf,
)
@info "  $db_path written"
# Write /index to database.h5
h5open(db_path, "r+") do f
    gr = HDF5.create_group(f, "index")
    write(gr, "phase_ids", index_phase_ids)
    write(gr, "phase_type", index_phase_type)
    write(gr, "station_idx", index_station_idx)
    write(gr, "distance", index_distance)
    write(gr, "azimuth", index_azimuth)
    write(gr, "greens_depth_idx", index_greens_depth_idx)
end
@info "  /index written ($n_phases phases)"


# 9. Write status_0.h5

@info "Writing status_0.h5 ..."

init_weights = fill(1.0 / n_misfit_modules, n_misfit_modules)

strategy = IO.Strategy(
    Float64(grid.strike0),
    Float64(grid.dstrike),
    Int32(grid.nstrike),
    Float64(grid.dip0),
    Float64(grid.ddip),
    Int32(grid.ndip),
    Float64(grid.rake0),
    Float64(grid.drake),
    Int32(grid.nrake),
    Int32.(1:n_depths),
    Int32.(1:n_bands),
    ones(Int32, n_phases),
    ones(Int32, n_stations),
    ones(Int32, n_stations),
    Float64.(init_weights),
    Float64[grid.strike0, grid.dip0, grid.rake0],
    Int32(1),
    Inf,
    Int32(0),
    Int32(0),
    "",
    zeros(Float64, n_bands, 3),
    zeros(Float64, n_bands, 0),
    zeros(Float64, n_depths),
)

status0_path = joinpath(data_dir, "status_0.h5")
h5open(status0_path, "w") do f
end
IO.write_strategy(status0_path, strategy)
@info "  $status0_path written"

# Summary

@info ""
@info "Stage complete:"
@info "  $(basename(db_path)) : /station ($n_stations rows)"
@info "  $(basename(db_path)) : /channel ($(length(channel_data)) channels)"
@info "  $(basename(db_path)) : /gf ($(length(gf_data)) depths)"
@info "  $(basename(db_path)) : /xcorr ($(length(xcorr_obs)) obs bands)"
@info "  $(basename(db_path)) : /polarity ($(length(polarity_obs)) channels)"
@info "  $(basename(db_path)) : /config, /event"
@info "  $(basename(status0_path)) : /strategy (initial grid, no trials)"
@info "  Stations: $n_stations | Phases: $n_phases | Depths: $n_depths | Bands: $n_bands"
@info ""
@info "="^70
