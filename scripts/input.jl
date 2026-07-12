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

# Validate: every station must have a corresponding phase pick (picks/stations
# ordering may differ; mismatched entries would silently misalign /station fields)
missing_picks = [s.id for s in stations if !haskey(station_to_idx, s.id)]
if !isempty(missing_picks)
    error("Stations without phase picks: $missing_picks")
end

# Build phase list: for each station, create P and S phase entries
# phase_list entries: (phase_id, phasetype, station_idx)

ch_map = Dict("N" => 1, "E" => 2, "Z" => 3)  # GF channel order [N, E, D]

function _channel_from_pid(pid::String)::String
    parts = split(pid, ".")
    return length(parts) >= 3 ? parts[3] : "Z"
end

phase_list = Tuple{String, String, Int}[]
for (si, s) in enumerate(stations)
    pick = picks[station_to_idx[s.id]]
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

# Per-type metadata for xcorrP / xcorrS
phases_P = [(pid, si) for (pid, pt, si) in phase_list if pt == "P"]
xcorrP_channel_id = [
    let s = stations[si]
        "$(s.network).$(s.station).$(s.channel)"
    end for (_, si) in phases_P
]
xcorrP_station_idx = Int32[si for (_, si) in phases_P]

phases_S = [(pid, si) for (pid, pt, si) in phase_list if pt == "S"]
xcorrS_channel_id = [
    let s = stations[si]
        "$(s.network).$(s.station).$(s.channel)"
    end for (_, si) in phases_S
]
xcorrS_station_idx = Int32[si for (_, si) in phases_S]

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
station_dict["P_time"] = [picks[station_to_idx[s.id]].P_time for s in stations]
station_dict["S_time"] = [picks[station_to_idx[s.id]].S_time for s in stations]
station_dict["P_polarity"] = [picks[station_to_idx[s.id]].P_polarity for s in stations]

@info "  station dict built ($n_stations stations)"

# 4. Build /channel — raw waveforms per channel
# NOTE: /channel stores raw (un-preprocessed) waveforms for verification/debugging only.
# These do NOT participate in forward misfit computation. The forward stage consumes
# the preprocessed products under /xcorrP/, /xcorrS/ (cross-correlation obs+gf, synamp = gf'@gf
# auto-correlation matrices) and /polarity (obs + gf_pol).

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

# 6. Preprocess waveforms — build /xcorrP/obs, /xcorrP/gf, /xcorrS/obs, /xcorrS/gf, /polarity/obs, /polarity/gf

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
polarity_channel_id = String[]
polarity_station_idx = Int32[]

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
        # gf preprocessed independently per depth (previously all depths reused depths[1])
        gf_lists =
            Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)

        for (pid, si) in phases_pt
            s = stations[si]
            dt = s.dt
            pick = picks[station_to_idx[s.id]]
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

            # Load GF for all depths — skip phase if any depth is missing (keeps obs/gf aligned)
            gf_per_depth = Dict{Float64, Matrix{Float64}}()
            all_gf_ok = true
            for depth_val in depths
                gf_full = get(gf_data[depth_val], ch_id, nothing)
                if gf_full === nothing
                    @warn "No GF for $ch_id at depth $depth_val, skipping preprocessing for $pid"
                    all_gf_ok = false
                    break
                end
                gf_per_depth[depth_val] = gf_full
            end
            if !all_gf_ok
                continue
            end

            # XCorr preprocessing — obs is depth-independent (preprocessed once with depths[1] GF);
            # GF is preprocessed per depth so filtering/window reflects each depth's GF.
            if "XCorr" in misfit_modules
                obs_proc, gf_proc0, _, obs_n2 = Signal.preprocess_xcorr!(
                    wf,
                    gf_per_depth[depths[1]],
                    dt,
                    arrival_sample,
                    low_cut,
                    high_cut,
                    window_factor;
                    filter_order = filter_order,
                )
                push!(obs_list, obs_proc)
                push!(obs_norm2_list, obs_n2)
                push!(gf_lists[depths[1]], gf_proc0)
                for depth_val in depths[2:end]
                    _, gf_proc_d, _, _ = Signal.preprocess_xcorr!(
                        wf,
                        gf_per_depth[depth_val],
                        dt,
                        arrival_sample,
                        low_cut,
                        high_cut,
                        window_factor;
                        filter_order = filter_order,
                    )
                    push!(gf_lists[depth_val], gf_proc_d)
                end
            end

            # Polarity preprocessing (P-wave only, first band only — polarity is frequency-independent)
            if "Polarity" in misfit_modules && ptype == "P" && freq_idx == 1
                obs_pol_val = Float64(pick.P_polarity)
                if obs_pol_val == -128.0
                    obs_pol_val = NaN
                end
                push!(polarity_obs, obs_pol_val)
                push!(polarity_channel_id, ch_id)
                push!(polarity_station_idx, Int32(si))
                for depth_val in depths
                    gf_pol, _ = Signal.preprocess_polarity!(
                        gf_per_depth[depth_val],
                        dt,
                        arrival_sample,
                        t_source,
                        pick.P_polarity,
                    )
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
            end
            obs_mat = zeros(Float64, np, nt_xc)
            for i in 1:np
                obs_mat[i, :] = obs_list[i]
            end
            xcorr_obs[key] = IO.XCorrObs(obs_mat, obs_norm2_list)
            # /xcorrP/gf/{depth}/{band} or /xcorrS/gf/{depth}/{band} — each depth stacked from its own preprocessed gf_proc
            for depth_val in depths
                gfl = gf_lists[depth_val]
                for i in 1:length(gfl)
                    gfl[i] = gfl[i][1:nt_xc, :]
                end
                gf_arr = zeros(Float64, np, 6, nt_xc)
                synamp_arr = zeros(Float64, np, 6, 6)
                for i in 1:np
                    gf_arr[i, :, :] = gfl[i]'
                    synamp_arr[i, :, :] = gfl[i]' * gfl[i]
                end
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
# Write phase metadata into xcorrP / xcorrS / polarity groups
h5open(db_path, "r+") do f
    pgr = f["xcorrP"]
    write(pgr, "channel_id", xcorrP_channel_id)
    write(pgr, "station_idx", xcorrP_station_idx)

    sgr = f["xcorrS"]
    write(sgr, "channel_id", xcorrS_channel_id)
    write(sgr, "station_idx", xcorrS_station_idx)

    if "Polarity" in misfit_modules
        polgr = f["polarity"]
        write(polgr, "channel_id", polarity_channel_id)
        write(polgr, "station_idx", polarity_station_idx)
    end
end
@info "  phase metadata written ($(length(xcorrP_channel_id)) P, $(length(xcorrS_channel_id)) S, $(length(polarity_channel_id)) polarity channels)"


# 9. Write status_0.h5

@info "Writing status_0.h5 ..."

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
    Int32(0),
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
@info "  $(basename(db_path)) : /xcorrP, /xcorrS ($(length(xcorr_obs)) obs bands)"
@info "  $(basename(db_path)) : /polarity ($(length(polarity_obs)) channels)"
@info "  $(basename(db_path)) : /config, /event"
@info "  $(basename(status0_path)) : /strategy (initial grid, no trials)"
@info "  Stations: $n_stations | Phases: $n_phases | Depths: $n_depths | Bands: $n_bands"
@info ""
@info "="^70
