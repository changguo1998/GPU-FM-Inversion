#!/usr/bin/env julia
#
# report.jl - Human-readable Markdown report generation.
#
# Reads the compact result.toml written by output.jl and renders report.md.
# No waveform, Green's function, or inversion-stage data is recomputed.
#
# Usage:
#   DATA_DIR=<dir> julia scripts/report.jl
#   julia scripts/report.jl <dir>

using Printf
using TOML

function resolve_data_dir()::String
    if haskey(ENV, "DATA_DIR")
        return abspath(ENV["DATA_DIR"])
    elseif !isempty(ARGS)
        return abspath(ARGS[1])
    end
    error("Usage: DATA_DIR=<dir> julia scripts/report.jl [<dir>]")
end

"""Format scalar values for compact Markdown cells."""
function format_value(value)::String
    value === nothing && return "N/A"
    if value isa AbstractFloat
        return isfinite(value) ? @sprintf("%.6g", value) : "N/A"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractVector
        return "[" * join(format_value.(value), ", ") * "]"
    end
    return string(value)
end

"""Escape values that could break a Markdown table."""
function markdown_cell(value)::String
    replace(format_value(value), "|" => "\\|", "\n" => " ")
end

"""Return a dictionary value or a default when a field is absent."""
function field(dict, key::String, default = nothing)
    haskey(dict, key) ? dict[key] : default
end

"""Read one cell from the row-oriented misfit matrix in result.toml."""
function matrix_cell(matrix, row::Int, column::Int)
    matrix isa AbstractVector || return nothing
    row <= length(matrix) || return nothing
    values = matrix[row]
    values isa AbstractVector || return nothing
    column <= length(values) ? values[column] : nothing
end

"""Append a two-column key/value Markdown table."""
function append_key_value_table!(io, rows)
    println(io, "| Parameter | Value |")
    println(io, "|---|---:|")
    for (label, value) in rows
        println(io, "| ", label, " | ", markdown_cell(value), " |")
    end
    println(io)
end

data_dir = resolve_data_dir()
result_path = joinpath(data_dir, "result.toml")
isfile(result_path) || error("result.toml not found: $result_path (run output.jl first)")
result = TOML.parsefile(result_path)

solution = field(result, "solution", Dict{String, Any}())
uncertainty = field(result, "uncertainty", Dict{String, Any}())
summary = field(result, "summary", Dict{String, Any}())
per_phase = field(result, "per_phase", Dict{String, Any}())
per_station = field(result, "per_station_summary", Dict{String, Any}())

io = IOBuffer()
println(io, "# 震源机制反演报告")
println(io)
println(io, "本报告由 `result.toml` 自动生成，仅整理最终结果，不重新计算反演数据。")
println(io)

println(io, "## 总结")
append_key_value_table!(
    io,
    [
        ("结果格式版本", field(result, "format_version")),
        ("总迭代次数", field(summary, "total_iterations")),
        ("总试次数", field(summary, "total_trials")),
        ("收敛原因", field(summary, "convergence_reason")),
    ],
)

println(io, "## 最优解")
append_key_value_table!(
    io,
    [
        ("Strike (deg)", field(solution, "strike")),
        ("Dip (deg)", field(solution, "dip")),
        ("Rake (deg)", field(solution, "rake")),
        ("Depth (km)", field(solution, "depth")),
        ("Duration (s)", field(solution, "duration")),
        ("Frequency index", field(solution, "freq_idx")),
        ("Duration index", field(solution, "duration_idx")),
        ("Misfit", field(solution, "misfit")),
    ],
)

moment_tensor = field(solution, "moment_tensor", nothing)
println(io, "Moment tensor (NED):")
println(io)
println(io, "```text")
println(io, format_value(moment_tensor))
println(io, "```")
println(io)

println(io, "## 不确定度")
append_key_value_table!(
    io,
    [
        ("Strike std (deg)", field(uncertainty, "strike_std")),
        ("Dip std (deg)", field(uncertainty, "dip_std")),
        ("Rake std (deg)", field(uncertainty, "rake_std")),
        ("Depth range (km)", field(uncertainty, "depth_range")),
        ("Frequency test misfit curve", field(uncertainty, "freq_test_misfit_curve")),
    ],
)

phase_ids = field(per_phase, "phase_id", Any[])
station_ids = field(per_phase, "station_id", Any[])
phase_types = field(per_phase, "phase_type", Any[])
cross_correlations = field(per_phase, "cross_correlation", Any[])
misfit_matrix = field(per_phase, "misfit_per_module", Any[])
misfit_modules = field(per_phase, "misfit_modules", Any[])
n_phases = maximum((
    length(phase_ids),
    length(station_ids),
    length(phase_types),
    length(cross_correlations),
    0,
))
n_modules = misfit_matrix isa AbstractVector ? length(misfit_matrix) : 0

println(io, "## Phase 质量")
if n_phases == 0
    println(io, "无 phase 结果。")
    println(io)
else
    headers = ["Phase", "Station", "Type", "Cross-correlation"]
    append!(
        headers,
        [
            i <= length(misfit_modules) ? "Misfit $(misfit_modules[i])" : "Misfit $(i)" for
            i in 1:n_modules
        ],
    )
    println(io, "| ", join(headers, " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for i in 1:n_phases
        cells = [
            i <= length(phase_ids) ? phase_ids[i] : nothing,
            i <= length(station_ids) ? station_ids[i] : nothing,
            i <= length(phase_types) ? phase_types[i] : nothing,
            i <= length(cross_correlations) ? cross_correlations[i] : nothing,
        ]
        append!(cells, [matrix_cell(misfit_matrix, j, i) for j in 1:n_modules])
        println(io, "| ", join(markdown_cell.(cells), " | "), " |")
    end
    println(io)
end

station_ids = field(per_station, "station_id", Any[])
n_station_phases = field(per_station, "n_phases", Any[])
station_cc = field(per_station, "mean_cross_correlation", Any[])
station_misfit_matrix = field(per_station, "misfit_per_module", Any[])
station_misfit_modules = field(per_station, "misfit_modules", Any[])
station_misfit = field(per_station, "misfit_total", Any[])
station_module_count = station_misfit_matrix isa AbstractVector ? length(station_misfit_matrix) : 0
n_stations = maximum((
    length(station_ids),
    length(n_station_phases),
    length(station_cc),
    length(station_misfit),
    0,
))

println(io, "## 台站汇总")
if n_stations == 0
    println(io, "无台站汇总结果。")
    println(io)
else
    headers = ["Station", "Phases", "Mean cross-correlation"]
    append!(
        headers,
        [
            i <= length(station_misfit_modules) ? "Misfit $(station_misfit_modules[i])" :
            "Misfit $(i)" for i in 1:station_module_count
        ],
    )
    push!(headers, "Total misfit")
    println(io, "| ", join(headers, " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for i in 1:n_stations
        cells = [
            i <= length(station_ids) ? station_ids[i] : nothing,
            i <= length(n_station_phases) ? n_station_phases[i] : nothing,
            i <= length(station_cc) ? station_cc[i] : nothing,
        ]
        append!(cells, [matrix_cell(station_misfit_matrix, j, i) for j in 1:station_module_count])
        push!(cells, i <= length(station_misfit) ? station_misfit[i] : nothing)
        println(io, "| ", join(markdown_cell.(cells), " | "), " |")
    end
    println(io)
end

println(io, "## 说明")
println(io, "- `N/A` 表示字段缺失、值为 NaN，或该统计量尚未实现。")
println(io, "- Phase misfit 列按 `result.toml` 中的模块顺序排列。")
println(io, "- 台站 misfit 列为未归一化、未加权的目标计算值。")

report_path = joinpath(data_dir, "report.md")
open(report_path, "w") do file
    write(file, String(take!(io)))
end
println("wrote $report_path")
