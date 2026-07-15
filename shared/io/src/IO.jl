module IO

using HDF5
using Dates

# Type Structs

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
    channel::String
    latitude::Float64
    longitude::Float64
    elevation::Float64
    dt::Float64
    begin_time::String
end

# Helper structs for flat-schema write_database

struct ModuleData
    # Per-band observation data
    obs::Dict{String, Matrix{Float64}}               # band_key -> obs matrix
    obs_norm2::Dict{String, Vector{Float64}}         # band_key -> norm2 vector
    # Per-depth, per-band GF data
    gf::Dict{Float64, Dict{String, Array{Float64, 3}}}   # depth -> band_key -> gf
    synamp::Dict{Float64, Dict{String, Array{Float64, 3}}}  # depth -> band_key -> synamp
    # Phase metadata
    channel_id::Vector{String}
    station_idx::Vector{Int32}
end

# Keyword constructor — obs_norm2/synamp default to empty for Polarity-style
function ModuleData(;
    obs::Dict{String, Matrix{Float64}},
    obs_norm2::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}(),
    gf::Dict{Float64, Dict{String, Array{Float64, 3}}},
    synamp::Dict{Float64, Dict{String, Array{Float64, 3}}} = Dict{
        Float64,
        Dict{String, Array{Float64, 3}},
    }(),
    channel_id::Vector{String} = String[],
    station_idx::Vector{Int32} = Int32[],
)
    :ModuleData
    return ModuleData(obs, obs_norm2, gf, synamp, channel_id, station_idx)
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
    depth_indices::Vector{Int32}
    freq_indices::Vector{Int32}
    iteration::Int32
end


# Exports

export EventInfo, StationInfo, PhasePick, TrialSet, Strategy, ModuleData
export h5create_group, h5exists
export read_config, read_event, read_phase_picks, read_stations
export read_waveform, read_trials, read_strategy, read_misfits
export read_greens
export write_database, write_trials, write_misfits, write_strategy, write_output
export write_paraspace, read_paraspace
export _read_group_recursive, _write_group_recursive
export parse_time_iso, haversine_distance, compute_azimuth
export extract_station, extract_phase_type
export find_latest_status

# Helpers

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

# Readers

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
            chans = [String(x) for x in read(gr["channel"])]
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
                    chans[i],
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
    h5open(f -> begin
        gr = f["strategy"]
        fi = if haskey(gr, "freq_indices")
            read(gr["freq_indices"])
        else
            Int32[]
        end
        Strategy(read(gr["depth_indices"]), fi, read(gr["iteration"]))
    end, h5file, "r")
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
    # New schema: /gf/{depth_val}/{channel_id}
    # Extract channel_id from phase_id (e.g. "NET.ST1.Z.P" -> "NET.ST1.Z")
    parts = split(phase_id, ".")
    ch_id = join(parts[1:3], ".")
    # Read depth from paraspace to map index -> depth value
    ps = read_paraspace(h5file)
    depth_vals = ps["depth"]
    depth_val = depth_vals[depth_idx]
    gf_path = "/gf/$(depth_val)/$(ch_id)"
    return h5open(f -> read(f[gf_path]), h5file, "r")
end


# Recursive write helper

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

# Writers

function write_database(
    h5file,
    config,
    event,
    station,
    channel_data,
    gf_data,
    module_data::Dict{String, ModuleData};
    paraspace = nothing,
)
    h5open(h5file, "w") do f
        # /paraspace - expanded parameter-space float arrays
        if paraspace !== nothing
            psgr = HDF5.create_group(f, "paraspace")
            for (k, v) in paraspace
                write(psgr, string(k), v)
            end
        end

        # /config - recursive write
        cfggr = HDF5.create_group(f, "config")
        _write_group_recursive(cfggr, config)

        # /event - scalar datasets
        evgr = HDF5.create_group(f, "event")
        for (k, v) in event
            write(evgr, string(k), v)
        end

        # /station - flat arrays
        stgr = HDF5.create_group(f, "station")
        for (k, v) in station
            write(stgr, string(k), v)
        end

        # /channel - raw waveforms
        chgr = HDF5.create_group(f, "channel")
        for (ch_id, wf) in channel_data
            write(chgr, ch_id, wf)
        end

        # /gf/{depth}/{channel_id}
        gfgr = HDF5.create_group(f, "gf")
        for (depth, ch_data) in gf_data
            dgr = HDF5.create_group(gfgr, string(depth))
            for (ch_id, gf_mat) in ch_data
                write(dgr, ch_id, gf_mat)
            end
        end

        # /{ModuleName}/ - iterate over all misfit module instances
        for mod_name in sort(collect(keys(module_data)))
            md = module_data[mod_name]
            m_gr = HDF5.create_group(f, mod_name)
            # Write phase metadata
            if !isempty(md.channel_id)
                write(m_gr, "channel_id", md.channel_id)
                write(m_gr, "station_idx", md.station_idx)
            end
            # Write per-band observation data
            obs_gr = HDF5.create_group(m_gr, "obs")
            for band_key in sort(collect(keys(md.obs)))
                obs_mat = md.obs[band_key]
                b_gr = HDF5.create_group(obs_gr, band_key)
                write(b_gr, "obs", obs_mat)
                if haskey(md.obs_norm2, band_key)
                    write(b_gr, "obs_norm2", md.obs_norm2[band_key])
                end
            end
            # Write per-depth, per-band GF data
            gf_gr = HDF5.create_group(m_gr, "gf")
            for depth in sort(collect(keys(md.gf)))
                bands = md.gf[depth]
                d_gr = HDF5.create_group(gf_gr, string(depth))
                for band_key in sort(collect(keys(bands)))
                    gf_arr = bands[band_key]
                    b_gr = HDF5.create_group(d_gr, band_key)
                    write(b_gr, "gf", gf_arr)
                    if haskey(md.synamp, depth) && haskey(md.synamp[depth], band_key)
                        write(b_gr, "synamp", md.synamp[depth][band_key])
                    end
                end
            end
        end
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
        write(gr, "N_trials", Int32(length(trials.strike)))
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
        write(gr, "depth_indices", strategy.depth_indices)
        write(gr, "freq_indices", strategy.freq_indices)
        write(gr, "iteration", strategy.iteration)
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
function parse_time_iso(t_str::String)::Float64
    isempty(t_str) && return NaN
    return datetime2unix(DateTime(t_str))
end

"""
    haversine_distance(lat1, lon1, lat2, lon2) -> Float64

Compute great-circle distance (km) between two points on a sphere
(Earth radius = 6371 km).
"""
function haversine_distance(lat1, lon1, lat2, lon2)::Float64
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
function compute_azimuth(lat1, lon1, lat2, lon2)::Float64
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
function extract_station(phase_id::String)::String
    parts = split(phase_id, '.')
    return join(parts[1:2], '.')
end

"""
    extract_phase_type(phase_id::String) -> String

Extract phase type from phase identifier.
"NET.ST1.Z.P" → "P"
"""
function extract_phase_type(phase_id::String)::String
    parts = split(phase_id, '.')
    return parts[end]
end

"""
    find_latest_status(status_dir::String) -> (filepath::String, iteration::Int)

Find the highest-numbered `status_N.h5` file in a directory.
Returns `(full_path, N)` or errors if none found.
"""
function find_latest_status(status_dir::String)::Tuple{String, Int}
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

# ── Paraspace ────────────────────────────────────────────────────────────

"""
    write_paraspace(h5file, paraspace::Dict)

Write `/paraspace` group, replacing any existing group.
Stores expanded float arrays for parameter-space dimensions:
- strike, dip, rake   (from grid expansion)
- depth_vals          (depth levels)
- frequency           (flat array: [low1, high1, low2, high2, ...])
"""
function write_paraspace(h5file, paraspace::Dict)
    h5open(h5file, "r+") do f
        if haskey(f, "paraspace")
            HDF5.delete_object(f["paraspace"])
        end
        gr = HDF5.create_group(f, "paraspace")
        for (k, v) in paraspace
            write(gr, string(k), v)
        end
    end
end

"""
    read_paraspace(h5file) -> Dict{String, Any}

Read `/paraspace` group into a Dict. Each key maps a parameter-space
name to its float array.
"""
function read_paraspace(h5file)::Dict{String, Any}
    return h5open(f -> _read_group_recursive(f["paraspace"]), h5file, "r")
end

end # module
