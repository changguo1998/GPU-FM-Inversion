module IO

using HDF5
using Dates

# ─────────────────────────────────────────────────────────
# Type Structs
# ─────────────────────────────────────────────────────────

struct EventInfo
    longitude::Float64
    latitude::Float64
    depth::Float64
    magnitude::Float64
    origintime::String
end

struct StationInfo
    id::String
    network::String
    station::String
    component::String
    latitude::Float64
    longitude::Float64
    elevation::Float64
    dt::Float64
    begin_time::String
end

struct PhasePick
    station_id::String
    P_time::String
    S_time::String
    P_polarity::Int8
end

struct TrialSet
    strike::Vector{Float64}
    dip::Vector{Float64}
    rake::Vector{Float64}
    depth::Vector{Float64}
    depth_idx::Vector{Int32}
    freq_idx::Vector{Int32}
end

struct Strategy
    strike0::Float64
    dstrike::Float64
    nstrike::Int32
    dip0::Float64
    ddip::Float64
    ndip::Int32
    rake0::Float64
    drake::Float64
    nrake::Int32
    depth_indices::Vector{Int32}
    freq_indices::Vector{Int32}
    xcorr_phase_mask::Vector{Int32}
    polarity_channel_mask::Vector{Int32}
    psr_channel_mask::Vector{Int32}
    module_weights::Vector{Float64}
    best_sdr::Vector{Float64}
    best_depth_index::Int32
    best_misfit::Float64
    iteration::Int32
    converged::Int32
    convergence_reason::String
    freq_accumulated::Matrix{Float64}
    freq_misfit_curve::Matrix{Float64}
    depth_misfit_accumulated::Vector{Float64}
end

struct Index
    phase_ids::Vector{String}
    phase_type::Vector{String}
    station_idx::Vector{Int32}
    distance::Vector{Float64}
    azimuth::Vector{Float64}
    greens_depth_idx::Matrix{Int32}
end

# ─────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────

export EventInfo, StationInfo, PhasePick, TrialSet, Strategy, Index
export h5create_group, h5exists
export read_config, read_event, read_phase_picks, read_stations
export read_waveform, read_trials, read_strategy, read_misfits
export read_greens, read_index
export write_database, write_trials, write_misfits, write_strategy, write_output
export _read_group_recursive, _write_group_recursive
export parse_time_iso, haversine_distance, compute_azimuth
export extract_station, extract_phase_type
export find_latest_status

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

"""
    h5create_group(h5file, path)

Create an HDF5 group at `path`, creating intermediate groups as needed.
"""
function h5create_group(h5file, path)
    h5open(f -> begin
        parts = split(path, '/'; keepempty = false)
        node = f
        for p in parts
            if haskey(node, p) && isgroup(node[p])
                node = node[p]
            else
                node = HDF5.create_group(node, p)
            end
        end
        node
    end, h5file, "r+")
end

"""
    h5exists(h5file, path)::Bool

Check whether a group or dataset exists at `path`.
"""
function h5exists(h5file, path)::Bool
    h5open(f -> begin
        parts = split(path, '/'; keepempty = false)
        node = f
        for p in parts
            if !haskey(node, p)
                return false
            end
            node = node[p]
        end
        return true
    end, h5file, "r")
end

"""
    _read_group_recursive(gr::HDF5.Group)::Dict{String,Any}

Recursively read an HDF5 group into a nested `Dict{String,Any}`.
Datasets become their values; subgroups become nested Dicts.
"""
function _read_group_recursive(gr)::Dict{String, Any}
    result = Dict{String, Any}()
    for name in keys(gr)
        obj = gr[name]
        if isa(obj, HDF5.Dataset)
            result[name] = read(obj)
        elseif isa(obj, HDF5.Group)
            result[name] = _read_group_recursive(obj)
        end
    end
    return result
end

"""
    read_config(h5file)::Dict{String,Any}

Recursively read `/config` group into a nested Dict.
"""
function read_config(h5file)::Dict{String, Any}
    return h5open(f -> _read_group_recursive(f["config"]), h5file, "r")
end

# ─────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────

function read_event(h5file)::EventInfo
    h5open(
        f -> begin
            gr = f["event"]
            EventInfo(
                read(gr["longitude"]),
                read(gr["latitude"]),
                read(gr["depth"]),
                read(gr["magnitude"]),
                String(read(gr["origintime"])),
            )
        end,
        h5file,
        "r",
    )
end

function read_phase_picks(h5file)::Vector{PhasePick}
    h5open(f -> begin
        gr = f["phase_picks"]
        ids = [String(x) for x in read(gr["station_ids"])]
        pt = [String(x) for x in read(gr["P_time"])]
        st = [String(x) for x in read(gr["S_time"])]
        pp = read(gr["P_polarity"])
        [PhasePick(ids[i], pt[i], st[i], pp[i]) for i in eachindex(ids)]
    end, h5file, "r")
end

function read_stations(h5file)::Vector{StationInfo}
    h5open(
        f -> begin
            gr = f["stations"]
            ids = [String(x) for x in read(gr["id"])]
            nets = [String(x) for x in read(gr["network"])]
            stas = [String(x) for x in read(gr["station"])]
            comps = [String(x) for x in read(gr["component"])]
            lats = read(gr["latitude"])
            lons = read(gr["longitude"])
            elevs = read(gr["elevation"])
            dts = read(gr["dt"])
            btimes = [String(x) for x in read(gr["begin_time"])]
            [
                StationInfo(
                    ids[i],
                    nets[i],
                    stas[i],
                    comps[i],
                    lats[i],
                    lons[i],
                    elevs[i],
                    dts[i],
                    btimes[i],
                ) for i in eachindex(ids)
            ]
        end,
        h5file,
        "r",
    )
end

function read_waveform(h5file, phase_id)::Vector{Float64}
    h5open(f -> read(f["waveforms/$(phase_id)"]), h5file, "r")
end

function read_trials(h5file)::TrialSet
    h5open(
        f -> begin
            gr = f["trials"]
            TrialSet(
                read(gr["strike"]),
                read(gr["dip"]),
                read(gr["rake"]),
                read(gr["depth"]),
                read(gr["depth_idx"]),
                read(gr["freq_idx"]),
            )
        end,
        h5file,
        "r",
    )
end

function read_strategy(h5file)::Strategy
    h5open(
        f -> begin
            gr = f["strategy"]
            Strategy(
                read(gr["strike0"]),
                read(gr["dstrike"]),
                read(gr["nstrike"]),
                read(gr["dip0"]),
                read(gr["ddip"]),
                read(gr["ndip"]),
                read(gr["rake0"]),
                read(gr["drake"]),
                read(gr["nrake"]),
                read(gr["depth_indices"]),
                read(gr["freq_indices"]),
                read(gr["xcorr_phase_mask"]),
                read(gr["polarity_channel_mask"]),
                read(gr["psr_channel_mask"]),
                read(gr["module_weights"]),
                read(gr["best_sdr"]),
                read(gr["best_depth_index"]),
                read(gr["best_misfit"]),
                read(gr["iteration"]),
                read(gr["converged"]),
                String(read(gr["convergence_reason"])),
                read(gr["freq_accumulated"]),
                read(gr["freq_misfit_curve"]),
                read(gr["depth_misfit_accumulated"]),
            )
        end,
        h5file,
        "r",
    )
end

function read_misfits(h5file)::Dict{Symbol, Matrix{Float64}}
    mis = Dict{Symbol, Matrix{Float64}}()
    h5open(f -> begin
        gr = f["misfits"]
        for name in keys(gr)
            mis[Symbol(name)] = read(gr[name])
        end
    end, h5file, "r")
    return mis
end

function read_greens(h5file, phase_id, depth_idx)::Matrix{Float64}
    h5open(f -> read(f["greens/$(phase_id)/$(depth_idx)"]), h5file, "r")
end

function read_index(h5file)::Index
    h5open(
        f -> begin
            gr = f["index"]
            Index(
                [String(x) for x in read(gr["phase_ids"])],
                [String(x) for x in read(gr["phase_type"])],
                read(gr["station_idx"]),
                read(gr["distance"]),
                read(gr["azimuth"]),
                read(gr["greens_depth_idx"]),
            )
        end,
        h5file,
        "r",
    )
end

# ─────────────────────────────────────────────────────────
# Recursive write helper
# ─────────────────────────────────────────────────────────

"""
    _write_group_recursive(parent::HDF5.Group, data::Dict)

Write nested Dict into HDF5 group recursively.
"""
function _write_group_recursive(parent, data)
    for (k, v) in data
        if v isa Dict
            sgr = HDF5.create_group(parent, k)
            _write_group_recursive(sgr, v)
        else
            write(parent, k, v)
        end
    end
end

# ─────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────

function write_database(h5file, greens, data, index, config)
    h5open(h5file, "w") do f
        # Write /index
        idxgr = HDF5.create_group(f, "index")
        write(idxgr, "phase_ids", index.phase_ids)
        write(idxgr, "phase_type", index.phase_type)
        write(idxgr, "station_idx", index.station_idx)
        write(idxgr, "distance", index.distance)
        write(idxgr, "azimuth", index.azimuth)
        write(idxgr, "greens_depth_idx", index.greens_depth_idx)

        # Write /greens — greens is Dict{String, Dict{Int32, Matrix{Float64}}}
        grgreens = HDF5.create_group(f, "greens")
        for (phase_id, depths) in greens
            pgr = HDF5.create_group(grgreens, phase_id)
            for (didx, mat) in depths
                write(pgr, string(didx), mat)
            end
        end

        # Write /data — data is Dict{Int, Dict{Symbol, Dict{String, ...}}}
        grdata = HDF5.create_group(f, "data")
        for (freq_idx, modules) in data
            fgr = HDF5.create_group(grdata, string(freq_idx))
            for (mod_name, phases) in modules
                mgrp = HDF5.create_group(fgr, string(mod_name))
                for (pid, contents) in phases
                    pgr = HDF5.create_group(mgrp, pid)
                    if contents isa Dict
                        for (k, v) in contents
                            write(pgr, k, v)
                        end
                    else
                        write(pgr, "data", contents)
                    end
                end
            end
        end

        # Write /config — recursive for arbitrary nesting
        cfggr = HDF5.create_group(f, "config")
        _write_group_recursive(cfggr, config)
    end
end

function write_trials(h5file, trials::TrialSet)
    h5open(h5file, "r+") do f
        if haskey(f, "trials")
            HDF5.delete_object(f["trials"])
        end
        gr = HDF5.create_group(f, "trials")
        write(gr, "strike", trials.strike)
        write(gr, "dip", trials.dip)
        write(gr, "rake", trials.rake)
        write(gr, "depth", trials.depth)
        write(gr, "depth_idx", trials.depth_idx)
        write(gr, "freq_idx", trials.freq_idx)
    end
end

"""
    write_misfits(h5file, modname::Symbol, data::AbstractArray)

Write misfit matrix for `modname` into `/misfits/{modname}`,
replacing any existing dataset.
"""
function write_misfits(h5file, modname::Symbol, data::AbstractArray)
    h5open(h5file, "r+") do f
        if !haskey(f, "misfits")
            HDF5.create_group(f, "misfits")
        end
        dsname = string(modname)
        if haskey(f["misfits"], dsname)
            HDF5.delete_object(f["misfits"][dsname])
        end
        write(f["misfits"], dsname, data)
    end
end

"""
    write_strategy(h5file, strategy::Strategy)

Write `/strategy` group, replacing any existing group.
Each stage writes complete datasets — no append mode.
"""
function write_strategy(h5file, strategy::Strategy)
    h5open(h5file, "r+") do f
        if haskey(f, "strategy")
            HDF5.delete_object(f["strategy"])
        end
        gr = HDF5.create_group(f, "strategy")
        write(gr, "strike0", strategy.strike0)
        write(gr, "dstrike", strategy.dstrike)
        write(gr, "nstrike", strategy.nstrike)
        write(gr, "dip0", strategy.dip0)
        write(gr, "ddip", strategy.ddip)
        write(gr, "ndip", strategy.ndip)
        write(gr, "rake0", strategy.rake0)
        write(gr, "drake", strategy.drake)
        write(gr, "nrake", strategy.nrake)
        write(gr, "depth_indices", strategy.depth_indices)
        write(gr, "freq_indices", strategy.freq_indices)
        write(gr, "xcorr_phase_mask", strategy.xcorr_phase_mask)
        write(gr, "polarity_channel_mask", strategy.polarity_channel_mask)
        write(gr, "psr_channel_mask", strategy.psr_channel_mask)
        write(gr, "module_weights", strategy.module_weights)
        write(gr, "best_sdr", strategy.best_sdr)
        write(gr, "best_depth_index", strategy.best_depth_index)
        write(gr, "best_misfit", strategy.best_misfit)
        write(gr, "iteration", strategy.iteration)
        write(gr, "converged", strategy.converged)
        write(gr, "convergence_reason", strategy.convergence_reason)
        write(gr, "freq_accumulated", strategy.freq_accumulated)
        write(gr, "freq_misfit_curve", strategy.freq_misfit_curve)
        write(gr, "depth_misfit_accumulated", strategy.depth_misfit_accumulated)
    end
end

function write_output(h5file, solution, uncertainty, per_phase, per_station_summary, summary)
    h5open(h5file, "w") do f
        # /solution
        solgr = HDF5.create_group(f, "solution")
        for (k, v) in solution
            write(solgr, k, v)
        end

        # /uncertainty
        ungr = HDF5.create_group(f, "uncertainty")
        for (k, v) in uncertainty
            write(ungr, k, v)
        end

        # /per_phase
        pphgr = HDF5.create_group(f, "per_phase")
        for (k, v) in per_phase
            write(pphgr, k, v)
        end

        # /per_station_summary
        pstgr = HDF5.create_group(f, "per_station_summary")
        for (k, v) in per_station_summary
            write(pstgr, k, v)
        end

        # /summary
        smgr = HDF5.create_group(f, "summary")
        for (k, v) in summary
            write(smgr, k, v)
        end
    end
end

"""
    parse_time_iso(t_str::String) -> Float64

Parse an ISO 8601 datetime string and return seconds since epoch.
Empty strings return NaN.
"""
function parse_time_iso(t_str::String)
    isempty(t_str) && return NaN
    return datetime2unix(DateTime(t_str))
end

"""
    haversine_distance(lat1, lon1, lat2, lon2) -> Float64

Compute great-circle distance (km) between two points on a sphere
(Earth radius = 6371 km).
"""
function haversine_distance(lat1, lon1, lat2, lon2)
    R = 6371.0
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    return 2 * R * asin(sqrt(a))
end

"""
    compute_azimuth(lat1, lon1, lat2, lon2) -> Float64

Compute azimuth (degrees, 0 = north, clockwise) from point 1 to point 2.
"""
function compute_azimuth(lat1, lon1, lat2, lon2)
    lat1r = deg2rad(lat1)
    lat2r = deg2rad(lat2)
    dlon = deg2rad(lon2 - lon1)
    x = sin(dlon) * cos(lat2r)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    az = rad2deg(atan(x, y))
    return mod(az, 360.0)
end

"""
    extract_station(phase_id::String) -> String

Extract station key from phase identifier.
"NET.ST1.Z.P" → "NET.ST1"
"""
function extract_station(phase_id::String)
    parts = split(phase_id, '.')
    return join(parts[1:2], '.')
end

"""
    extract_phase_type(phase_id::String) -> String

Extract phase type from phase identifier.
"NET.ST1.Z.P" → "P"
"""
function extract_phase_type(phase_id::String)
    parts = split(phase_id, '.')
    return parts[end]
end

"""
    find_latest_status(status_dir::String) -> (filepath::String, iteration::Int)

Find the highest-numbered `status_N.h5` file in a directory.
Returns `(full_path, N)` or errors if none found.
"""
function find_latest_status(status_dir::String)
    pattern = r"^status_(\d+)\.h5$"
    max_n = -1
    latest = ""
    for entry in readdir(status_dir; join = true)
        m = match(pattern, basename(entry))
        if m !== nothing
            n = parse(Int, m.captures[1])
            if n > max_n
                max_n = n
                latest = entry
            end
        end
    end
    if max_n == -1
        error("no status files found in $status_dir")
    end
    return (latest, max_n)
end
end # module
