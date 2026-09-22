# config.jl — SAC event example using SeisTools SAC I/O and DWN Green functions.
#
# This configuration is intentionally self-contained so it can be run with:
#   bash ../../driver.sh --data-dir examples/event1 --trial-budget 10000
#
# P/S picks are estimated from the event origin and the supplied 1-D model.
# They are starting picks, not analyst-reviewed phase picks.

using Dates
using Misfit
using SeisTools
using DWN

p_observed = observed(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
p_synthetic = synthetic(P; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
s_observed = observed(S; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)
s_synthetic = synthetic(S; band = (0.5, 2.0), window = (-2, 8), filter_order = 4)

Config.@objective XcorrP = 1 - maxCC(p_observed, p_synthetic; maxlag = 3)
Config.@objective XcorrS = 1 - maxCC(s_observed, s_synthetic; maxlag = 3)
Config.@objective LagP = lagCC(p_observed, p_synthetic; maxlag = 3)
Config.@objective LagS = lagCC(s_observed, s_synthetic; maxlag = 3)
Config.@objective Psr =
    abs2(log(rms(s_observed) / rms(p_observed)) - log(rms(s_synthetic) / rms(p_synthetic)))

Config.freq_bands() = [(0.5, 2.0)]
Config.depths() = [5.0, 9.0, 15.0]
Config.durations() = [0.1, 0.2, 0.3]
Config.phase_fields() = Dict("P" => :P_time, "S" => :S_time)
Config.polarity_fields() = Dict{String, Symbol}()

const _DIR = @__DIR__
const _DECIMATION = 5
const _SAC_DT = 0.01
const _DT = _SAC_DT * _DECIMATION
const _GF_NPTS = parse(Int, get(ENV, "EVENT1_GF_NPTS", "8192"))
const _GF_MAX_ORDER = parse(Int, get(ENV, "EVENT1_DWN_MAX_ORDER", "1000"))

function _read_event()
    fields = split(
        strip(
            first(
                filter(
                    x -> !isempty(strip(x)) && !startswith(strip(x), '#'),
                    readlines(joinpath(_DIR, "event.txt")),
                ),
            ),
        ),
    )
    origin = DateTime("$(replace(fields[1], '/' => '-'))T$(fields[2])")
    return (
        origin,
        parse(Float64, fields[3]),
        parse(Float64, fields[4]),
        parse(Float64, fields[5]),
        parse(Float64, fields[6]),
    )
end

const _EVENT_ORIGIN, _EVENT_LAT, _EVENT_LON, _EVENT_DEPTH, _EVENT_MAG = _read_event()

const _SAC_INDEX = let
    index = Dict{String, Tuple{String, Dict{String, Any}}}()
    for path in sort(readdir(joinpath(_DIR, "sac"); join = true))
        endswith(lowercase(path), ".sac") || continue
        header, data = SeisTools.SAC.read(path)
        component =
            get(Dict("BHE" => "E", "BHN" => "N", "BHZ" => "Z"), header["kcmpnm"], nothing)
        component === nothing && continue
        network = header["knetwk"]
        station = header["kstnm"]
        key = "$network.$station.$component"
        index[key] = (path, header)
    end
    isempty(index) && error("no SAC files found under $(_DIR)/sac")
    index
end

const _WAVE_CACHE = Dict{String, Vector{Float64}}()
const _STATION_KEYS = sort(unique(join(split(key, ".")[1:2], ".") for key in keys(_SAC_INDEX)))

function _station_header(station_key::String)
    key = first(filter(k -> startswith(k, "$station_key."), keys(_SAC_INDEX)))
    return _SAC_INDEX[key][2]
end

function _iso(dt::DateTime)
    return Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS.sss")
end

function _waveform(channel_key::String)
    if !haskey(_WAVE_CACHE, channel_key)
        path, _ = _SAC_INDEX[channel_key]
        _, data = SeisTools.SAC.read(path)
        _WAVE_CACHE[channel_key] = Float64.(data[1:_DECIMATION:end])
    end
    return _WAVE_CACHE[channel_key]
end

Config.load_event() =
    IO.EventInfo(_EVENT_LON, _EVENT_LAT, _EVENT_DEPTH, _EVENT_MAG, _iso(_EVENT_ORIGIN))

Config.load_stations() = begin
    stations = IO.StationInfo[]
    for station_key in _STATION_KEYS
        network, station = split(station_key, "."; limit = 2)
        header = _station_header(station_key)
        for component in ("E", "N", "Z")
            key = "$station_key.$component"
            haskey(_SAC_INDEX, key) || continue
            push!(
                stations,
                IO.StationInfo(
                    station_key,
                    network,
                    station,
                    component,
                    header["stla"],
                    header["stlo"],
                    header["stel"],
                    _DT,
                    _iso(
                        DateTime(
                            header["nzyear"],
                            1,
                            1,
                            header["nzhour"],
                            header["nzmin"],
                            header["nzsec"],
                            header["nzmsec"],
                        ) + Day(header["nzjday"] - 1),
                    ),
                ),
            )
        end
    end
    stations
end

function _model()
    rows = [
        parse.(Float64, split(strip(line))) for
        line in readlines(joinpath(_DIR, "model.txt")) if
        !isempty(strip(line)) && !startswith(strip(line), '#')
    ]
    matrix = reduce(vcat, (reshape(row, 1, :) for row in rows))
    positive = matrix[matrix[:, 1] .>= 0.0, :]
    boundaries = vcat(0.0, positive[:, 1], positive[end, 1] + 100.0)
    thickness = diff(boundaries)
    values = vcat(positive, positive[end:end, :])
    return hcat(thickness, values[:, 2:4], fill(1000.0, size(values, 1), 2))
end

const _DWN_MODEL = _model()
const _STATION_GFS = Dict{Float64, Dict{String, Array{Float64, 3}}}()

function _station_geometry()
    stations = Config.load_stations()
    unique_stations = Dict{String, IO.StationInfo}()
    for station in stations
        unique_stations[station.id] = station
    end
    ordered = sort(collect(values(unique_stations)); by = s -> s.id)
    receivers = Tuple{Float64, Float64}[]
    for station in ordered
        distance =
            IO.haversine_distance(_EVENT_LAT, _EVENT_LON, station.latitude, station.longitude)
        azimuth = IO.compute_azimuth(_EVENT_LAT, _EVENT_LON, station.latitude, station.longitude)
        push!(receivers, (max(distance, 0.01), azimuth))
    end
    return ordered, receivers
end

function _build_gfs(depth::Float64)
    stations, receivers = _station_geometry()
    spectra = DWN.dwn(_DWN_MODEL, depth, 0.1, receivers, 0.0, _GF_NPTS, _DT, _GF_MAX_ORDER)
    waveforms = DWN.freqspec2timeseries(spectra, _GF_NPTS)
    result = Dict{String, Array{Float64, 3}}()
    source_shift =
        round(Int, Dates.value(_EVENT_ORIGIN - DateTime(stations[1].begin_time)) / (1000.0 * _DT))
    for (station_index, station) in enumerate(stations)
        base = zeros(Float64, _GF_NPTS, 6, 3)
        for component in 1:3, moment in 1:6
            base[:, moment, component] = real.(waveforms[station_index, component, moment])
        end
        shifted = zeros(Float64, size(base))
        if source_shift < _GF_NPTS
            shifted[(source_shift + 1):end, :, :] = base[1:(end - source_shift), :, :]
        end
        shifted[:, :, 3] .*= -1.0
        for channel in ("E", "N", "Z")
            result["$(station.id).$channel"] = shifted
        end
    end
    return result
end

function _gfs_for_depth(depth::Float64)
    if !haskey(_STATION_GFS, depth)
        @info "DWN: calculating Green functions" depth = depth npts = _GF_NPTS dt = _DT
        _STATION_GFS[depth] = _build_gfs(depth)
    end
    return _STATION_GFS[depth]
end

Config.load_phase_picks() = begin
    picks = IO.PhasePick[]
    vp = _DWN_MODEL[1, 2]
    vs = _DWN_MODEL[1, 3]
    for station_key in _STATION_KEYS
        header = _station_header(station_key)
        distance = IO.haversine_distance(_EVENT_LAT, _EVENT_LON, header["stla"], header["stlo"])
        distance3d = sqrt(distance^2 + _EVENT_DEPTH^2)
        p_time = _iso(_EVENT_ORIGIN + Millisecond(round(Int, 1000 * distance3d / vp)))
        s_time = _iso(_EVENT_ORIGIN + Millisecond(round(Int, 1000 * distance3d / vs)))
        push!(picks, IO.PhasePick(station_key, p_time, s_time, Int8(0)))
    end
    picks
end

Config.load_waveform(phase_id::String) = begin
    parts = split(phase_id, ".")
    length(parts) == 4 || error("invalid phase id: $phase_id")
    _waveform(join(parts[1:3], "."))
end

Config.load_gf(src_lat, src_lon, src_depth, sta_lat, sta_lon) = begin
    station = first(
        filter(
            s ->
                isapprox(s.latitude, sta_lat; atol = 1e-5) &&
                isapprox(s.longitude, sta_lon; atol = 1e-5),
            Config.load_stations(),
        ),
    )
    (_gfs_for_depth(src_depth)["$(station.id).$(station.channel)"], _DT, 0.0, 0.0)
end
