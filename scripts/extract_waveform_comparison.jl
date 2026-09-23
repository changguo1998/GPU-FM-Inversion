#!/usr/bin/env julia

using HDF5
using LinearAlgebra
using Printf
using IO
using Signal

length(ARGS) == 4 || error(
    "usage: extract_waveform_comparison.jl <database.h5> <status.h5> <output.h5> <output-dir>",
)

database_path, status_path, result_path, output_dir = ARGS
mkpath(output_dir)

function best_trial_row(values, best_index, trial_count, channel_count, path)
    size(values) == (trial_count, channel_count) ||
        error("unexpected shape for $path: $(size(values))")
    return values[best_index, :]
end

function shift_synthetic(synthetic, lag)
    aligned = zeros(Float64, length(synthetic))
    for sample in eachindex(aligned)
        source = sample - lag
        checkbounds(Bool, synthetic, source) && (aligned[sample] = synthetic[source])
    end
    return aligned
end

function trim_aligned_waveforms(observed, synthetic, arrival_sample, pre_samples, post_samples, lag)
    window = (arrival_sample - pre_samples):(arrival_sample + post_samples)
    first(window) >= 1 && last(window) <= length(observed) ||
        error("observation window is outside the full waveform")
    last(window) <= length(synthetic) || error("synthetic window is outside the full waveform")
    aligned = shift_synthetic(synthetic, lag)
    return observed[window], aligned[window]
end

function normalized_trace(trace)
    scale = maximum(abs, trace; init = 0.0)
    return scale > 0.0 ? 0.75 .* trace ./ scale : zeros(Float64, length(trace))
end

function correlation(observed, synthetic)
    denominator = norm(observed) * norm(synthetic)
    return denominator > 0.0 ? dot(observed, synthetic) / denominator : 0.0
end

function component_name(channel_id)
    parts = split(channel_id, '.')
    isempty(parts) && error("invalid channel id: $channel_id")
    return parts[end]
end

total, depth_indices, frequency_indices, duration_indices = h5open(status_path, "r") do file
    (
        Float64.(read(file["/aggregate/total"])),
        Int.(read(file["/trials/depth_idx"])),
        Int.(read(file["/trials/freq_idx"])),
        Int.(read(file["/trials/duration_idx"])),
    )
end
trial_count = length(total)
best_index = argmin(total)
depth_index = depth_indices[best_index]
frequency_index = frequency_indices[best_index]
duration_index = duration_indices[best_index]

moment_tensor, result_misfit = h5open(result_path, "r") do file
    (Float64.(read(file["/solution/moment_tensor"])), Float64(read(file["/solution/misfit"])))
end
length(moment_tensor) == 6 || error("solution moment tensor must have six components")
isapprox(total[best_index], result_misfit; atol = 1e-10, rtol = 1e-10) ||
    error("best status misfit does not match output.h5")

component_order = ["Z", "N", "E"]
station_ids = String[]
station_dt = Float64[]
station_begin_time = String[]
panel_data = Dict{Tuple{Int, String}, Matrix{Float64}}()

h5open(database_path, "r") do database
    append!(station_ids, String.(read(database["/station/id"])))
    append!(station_dt, Float64.(read(database["/station/dt"])))
    append!(station_begin_time, String.(read(database["/station/begin_time"])))
    frequencies = Float64.(read(database["/paraspace/frequency"]))
    durations = Float64.(read(database["/paraspace/duration"]))
    duration = durations[duration_index]

    h5open(status_path, "r") do status
        for (phase, module_name) in (("P", "XcorrP"), ("S", "XcorrS"))
            module_path = "/$module_name"
            haskey(database, module_path) || error("database has no $module_path group")
            pick_times = String.(read(database["/station/$(phase)_time"]))
            channel_ids = String.(read(database["$module_path/channel_id"]))
            station_indices = Int.(read(database["$module_path/station_idx"]))
            channel_count = length(channel_ids)
            length(station_indices) == channel_count ||
                error("$module_name channel and station indices differ in length")

            saved_observed =
                Float64.(read(database["$module_path/obs/$frequency_index/obs"]))
            size(saved_observed, 1) == channel_count || error(
                "unexpected $module_name observation shape: $(size(saved_observed))",
            )

            lag_path = "/intermediates/$module_name/best_lag"
            cc_path = "/intermediates/$module_name/cc_max"
            lags = Int.(
                best_trial_row(
                    read(status[lag_path]),
                    best_index,
                    trial_count,
                    channel_count,
                    lag_path,
                ),
            )
            correlations = Float64.(
                best_trial_row(
                    read(status[cc_path]),
                    best_index,
                    trial_count,
                    channel_count,
                    cc_path,
                ),
            )

            trim = Float64.(read(database["/config/$module_name/trim"]))
            band_low_indices = Int.(read(database["/config/$module_name/band_low"]))
            band_high_indices = Int.(read(database["/config/$module_name/band_high"]))
            filter_order = Int(read(database["/config/$module_name/filter_order"]))
            low_frequency = frequencies[band_low_indices[frequency_index]]
            high_frequency = frequencies[band_high_indices[frequency_index]]
            sample_count = size(saved_observed, 2)

            for station_index in eachindex(station_ids)
                dt = station_dt[station_index]
                pre_samples = max(1, round(Int, abs(trim[1]) / high_frequency / dt))
                post_samples = max(1, round(Int, abs(trim[2]) / high_frequency / dt))
                pre_samples + post_samples + 1 == sample_count || error(
                    "$module_name window length does not match station $(station_ids[station_index])",
                )
                time = ((1:sample_count) .- (pre_samples + 1)) .* dt
                panel = fill(NaN, sample_count, 7)
                panel[:, 1] = time

                for (component_index, component) in enumerate(component_order)
                    matches = findall(
                        channel ->
                            station_indices[channel] == station_index &&
                            component_name(channel_ids[channel]) == component,
                        eachindex(channel_ids),
                    )
                    isempty(matches) && continue
                    length(matches) == 1 || error(
                        "$module_name has duplicate $component channels for $(station_ids[station_index])",
                    )
                    channel = only(matches)
                    channel_id = channel_ids[channel]
                    observed_full = Float64.(read(database["/channel/$channel_id"]))
                    greens_full = Float64.(read(database["/gf/$depth_index/$channel_id"]))
                    size(greens_full, 2) == 6 ||
                        error("unexpected Green function shape for $channel_id: $(size(greens_full))")
                    observed_full = Signal.preprocess_waveform!(
                        observed_full,
                        dt,
                        low_frequency,
                        high_frequency;
                        order = filter_order,
                    )
                    for column in axes(greens_full, 2)
                        greens_full[:, column] = Signal.convolve_gaussian_stf(
                            greens_full[:, column],
                            duration,
                            dt,
                        )
                        greens_full[:, column] = Signal.preprocess_waveform!(
                            greens_full[:, column],
                            dt,
                            low_frequency,
                            high_frequency;
                            order = filter_order,
                        )
                    end
                    synthetic_full = greens_full * moment_tensor
                    begin_time = IO.parse_time_iso(station_begin_time[station_index])
                    pick_time = IO.parse_time_iso(pick_times[station_index])
                    arrival_sample = if isnan(begin_time) || isnan(pick_time)
                        length(observed_full) ÷ 2
                    else
                        clamp(
                            round(Int, (pick_time - begin_time) / dt) + 1,
                            1,
                            length(observed_full),
                        )
                    end
                    obs, aligned = trim_aligned_waveforms(
                        observed_full,
                        synthetic_full,
                        arrival_sample,
                        pre_samples,
                        post_samples,
                        lags[channel],
                    )
                    isapprox(obs, vec(saved_observed[channel, :]); atol = 1e-12, rtol = 1e-12) ||
                        error("$module_name $channel_id observation preprocessing mismatch")
                    window = (arrival_sample - pre_samples):(arrival_sample + post_samples)
                    inversion_aligned = shift_synthetic(synthetic_full[window], lags[channel])
                    recovered_cc = correlation(obs, inversion_aligned)
                    isapprox(recovered_cc, correlations[channel]; atol = 1e-10, rtol = 1e-10) ||
                        error(
                            "$module_name $channel_id alignment mismatch: " *
                            "stored=$(correlations[channel]), recovered=$recovered_cc",
                        )

                    observed_column = 2 * component_index
                    synthetic_column = observed_column + 1
                    panel[:, observed_column] = normalized_trace(obs)
                    panel[:, synthetic_column] = normalized_trace(aligned)
                end
                panel_data[(station_index, phase)] = panel
            end
        end
    end
end

open(joinpath(output_dir, "stations.txt"), "w") do io
    for station_id in station_ids
        println(io, station_id)
    end
end

for station_index in eachindex(station_ids), phase in ("P", "S")
    path = joinpath(output_dir, @sprintf("%03d_%s.dat", station_index, phase))
    panel = panel_data[(station_index, phase)]
    open(path, "w") do io
        println(io, "# time obs_Z syn_Z obs_N syn_N obs_E syn_E")
        for row in axes(panel, 1)
            println(io, join(panel[row, :], '\t'))
        end
    end
end

@info "waveform comparison extracted" best_trial = best_index stations = length(station_ids)
