"""Sum channel-indexed and station-indexed objective values into trial misfits."""
function hierarchical_sum(
    objective_values::AbstractDict,
    channel_ids::AbstractDict,
    station_indices::AbstractDict,
    levels::AbstractDict,
    n_stations::Integer;
    absolute_modules = Set(),
)
    n_stations >= 0 || throw(ArgumentError("station count must be non-negative"))
    isempty(objective_values) && throw(ArgumentError("no objective values to aggregate"))

    channel_station = Dict{String, Int32}()
    for name in keys(objective_values)
        haskey(channel_ids, name) || continue
        ids = channel_ids[name]
        indices = station_indices[name]
        matrix = objective_values[name]
        size(matrix, 1) == length(ids) == length(indices) ||
            throw(DimensionMismatch("$name channel rows and indices differ"))
        for (channel_id, station_idx) in zip(ids, indices)
            1 <= station_idx <= n_stations ||
                throw(ArgumentError("$name station index $station_idx is out of range"))
            previous = get(channel_station, channel_id, station_idx)
            previous == station_idx ||
                throw(ArgumentError("channel $channel_id maps to multiple stations"))
            channel_station[channel_id] = station_idx
        end
    end

    ordered_channels = sort!(collect(keys(channel_station)))
    channel_row = Dict(channel_id => row for (row, channel_id) in enumerate(ordered_channels))
    n_trials = size(first(values(objective_values)), 2)
    channel_phase = zeros(Float64, length(ordered_channels), n_trials)
    channel_direct = zeros(Float64, length(ordered_channels), n_trials)
    station_direct = zeros(Float64, n_stations, n_trials)

    for (name, matrix) in objective_values
        size(matrix, 2) == n_trials || throw(DimensionMismatch("$name trial count differs"))
        level = get(levels, name, nothing)
        make_absolute = name in absolute_modules
        if haskey(channel_ids, name)
            level in (:phase, :channel) ||
                throw(ArgumentError("$name channel-indexed level must be :phase or :channel"))
            ids = channel_ids[name]
            target = level == :phase ? channel_phase : channel_direct
            for trial in axes(matrix, 2), row in axes(matrix, 1)
                value = matrix[row, trial]
                isfinite(value) || continue
                contribution = make_absolute ? abs(value) : value
                target[channel_row[ids[row]], trial] += contribution
            end
        else
            level == :station ||
                throw(ArgumentError("$name without channel indices must have :station level"))
            size(matrix, 1) == n_stations ||
                throw(DimensionMismatch("$name must be channel-indexed or station-indexed"))
            for trial in axes(matrix, 2), station in axes(matrix, 1)
                value = matrix[station, trial]
                isfinite(value) || continue
                contribution = make_absolute ? abs(value) : value
                station_direct[station, trial] += contribution
            end
        end
    end

    channel_total = channel_phase .+ channel_direct
    station_channel = zeros(Float64, n_stations, n_trials)
    for (row, channel_id) in enumerate(ordered_channels)
        station_channel[channel_station[channel_id], :] .+= channel_total[row, :]
    end
    station_total = station_channel .+ station_direct
    total = vec(sum(station_total; dims = 1))
    return (
        channel_id = ordered_channels,
        channel_station_idx = Int32[channel_station[id] for id in ordered_channels],
        channel_phase_sum = channel_phase,
        channel_direct_sum = channel_direct,
        channel_total = channel_total,
        station_channel_sum = station_channel,
        station_direct_sum = station_direct,
        station_total = station_total,
        total = total,
    )
end
