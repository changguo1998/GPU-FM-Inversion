#!/usr/bin/env julia
#
# output.jl - 输出编译
#
# 读最新 status_N.h5 的 /misfits + /trials 与 database.h5 元数据,
# 选择 best trial 并写 output.h5 (/solution, /uncertainty, /per_phase,
# /per_station_summary, /summary)。
#
# 简化说明 (TODO):
#   - best trial 由 XcorrP 主 misfit 决定 (无量纲, 各模块量纲不同,
#     跨模块加权聚合属 assess 未完成部分);
#   - /uncertainty.freq_test_misfit_curve 填 NaN (未实现);
#   - /summary.convergence_reason 固定 "single iteration (refinement pending)"。
#
# Usage:
#   DATA_DIR=<dir> julia scripts/output.jl

using HDF5
using Statistics

using StageLog

using IO, MT

data_dir = ENV["DATA_DIR"]
StageLog.setup_logger!("output", joinpath(data_dir, "output.log"))

db_path = joinpath(data_dir, "database.h5")
status_dir = joinpath(data_dir, "status")
out_path = joinpath(data_dir, "output.h5")

status_path, iter_n = IO.find_latest_status(status_dir)

@info "output stage started (status $status_path, iteration $iter_n)"

# === 1. 读 trials + misfits ===
trials = IO.read_trials(status_path)
misfits = IO.read_misfits(status_path)  # Dict{Symbol, Matrix{Float64}} [entries × trials]
modules = sort(collect(keys(misfits)))

n_trials = length(trials.strike)
xcorr_mods = [Symbol(m) for m in (:XcorrP, :XcorrS) if haskey(misfits, Symbol(m))]
# 无 Xcorr → 退化为任意模块
if isempty(xcorr_mods)
    xcorr_mods = collect(keys(misfits))
end

# === 2. best trial: Xcorr 主 misfit 每 trial 取 entries 均值后 argmin ===
"""对 misfit 矩阵 [entries × trials] 每列取均值 (NaN-safe)。"""
function column_means(m::Matrix{Float64})::Vector{Float64}
    out = fill(NaN, size(m, 2))
    for t in axes(m, 2)
        vals = m[:, t]
        out[t] = mean(filter(!isnan, vals))
    end
    return out
end

total = zeros(n_trials)
for m in xcorr_mods
    total .+= column_means(misfits[m]) ./ length(xcorr_mods)
end
best_idx = argmin(total)
best = (
    strike = trials.strike[best_idx],
    dip = trials.dip[best_idx],
    rake = trials.rake[best_idx],
    depth = trials.depth[best_idx],
    depth_idx = trials.depth_idx[best_idx],
    freq_idx = trials.freq_idx[best_idx],
    misfit = total[best_idx],
)
@info "best trial #$best_idx: SDR=$(best.strike),$(best.dip),$(best.rake) depth=$(best.depth) km, misfit=$(best.misfit)"

# === 3. solution ===
mt = MT.sdr_to_mt(best.strike, best.dip, best.rake)
solution = Dict{String, Any}(
    "strike" => best.strike,
    "dip" => best.dip,
    "rake" => best.rake,
    "depth" => best.depth,
    "freq_idx" => Float64(best.freq_idx),
    "moment_tensor" => mt,
    "misfit" => best.misfit,
)

# === 4. uncertainty: best 邻域 (misfit ≤ 1.05×best) SDR std ===
thr = best.misfit * 1.05
nb = findall(total .<= thr)
nb2 = length(nb) >= 2 ? nb : [best_idx]
uncertainty = Dict{String, Any}(
    "strike_std" => std(trials.strike[nb2]),
    "dip_std" => std(trials.dip[nb2]),
    "rake_std" => std(trials.rake[nb2]),
    "depth_range" => [minimum(trials.depth[nb2]), maximum(trials.depth[nb2])],
    "freq_test_misfit_curve" => fill(NaN, 1, 1),
)

# === 5. per_phase: XcorrP/XcorrS 通道 (best trial 各模块 entry misfit) ===
function module_has(mod_name::String, field::String)::Bool
    return h5open(f -> haskey(f, "/$mod_name/$field"), db_path, "r")
end

function read_module_field(mod_name::String, field::String)
    return h5open(f -> read(f["/$mod_name/$field"]), db_path, "r")
end

phase_ids = String[]
stations_phase = String[]
phase_types = String[]
for (mod_name, ptype) in ((:XcorrP, "P"), (:XcorrS, "S"))
    if haskey(misfits, mod_name)
        chs = read_module_field(string(mod_name), "channel_id")
        append!(phase_ids, chs)
        append!(phase_types, fill(ptype, length(chs)))
        append!(stations_phase, [join(split(c, ".")[1:2], ".") for c in chs])
    end
end
n_phases = length(phase_ids)

misfit_per_module = zeros(length(modules), n_phases)
stations = IO.read_stations(db_path)
for (ri, m) in enumerate(modules)
    if haskey(misfits, m)
        misfit_per_module[ri, :] .= NaN
        module_has(string(m), "channel_id") || continue
        mids = read_module_field(string(m), "channel_id")
        if length(mids) == size(misfits[m], 1)
            # 行 = channel: 按 channel 对齐 phase_ids
            for (pi, pid) in enumerate(phase_ids)
                ei = findfirst(==(pid), mids)
                ei !== nothing && (misfit_per_module[ri, pi] = misfits[m][ei, best_idx])
            end
        else
            # 行 = station (如 Polarity/RelShift)。unique 台站顺序与
            # forward kernel 的 station 轴一致; Polarity 行 1 是 C++ 0-based
            # dummy, 后续行 1:1 对应物理台站。
            uids = sort(unique(stations_phase))
            nrows = size(misfits[m], 1)
            for (pi, st) in enumerate(stations_phase)
                k = findfirst(==(st), uids)
                k === nothing && continue
                row = nrows == length(uids) + 1 ? k + 1 : k
                row <= nrows && (misfit_per_module[ri, pi] = misfits[m][row, best_idx])
            end
        end
    end
end

# cross_correlation: 从 /intermediates 读 XcorrP/S cc_max 在 best trial 的值
function read_cc_column(key::String, idx::Int)::Vector{Float64}
    # cc_max 读取为 [N_trials × N_phases] (C-order written); 取 best trial 行
    return h5open(f -> Float64.(read(f["/intermediates/$key/cc_max"])[idx, :]), status_path, "r")
end

cross_corr = fill(NaN, n_phases)
for (mod_name, ptype) in ((:XcorrP, "P"), (:XcorrS, "S"))
    if haskey(misfits, mod_name)
        chs = read_module_field(string(mod_name), "channel_id")
        col = read_cc_column(string(mod_name), best_idx)
        # cc_max 每列对应 channel rows; 当行列数与 channel_id 一致时按通道对齐
        if length(col) == length(chs)
            for (i, c) in enumerate(chs)
                pi = findfirst(==(c), phase_ids)
                pi !== nothing && (cross_corr[pi] = col[i])
            end
        end
    end
end

per_phase = Dict{String, Any}(
    "phase_id" => phase_ids,
    "station_id" => stations_phase,
    "phase_type" => phase_types,
    "misfit_per_module" => misfit_per_module,
    "selected" => ones(Int32, n_phases),
    "cross_correlation" => cross_corr,
)

# === 6. per_station_summary ===
uniq_stations = [s.id for s in stations]
n_st_cross = [0.0 for _ in uniq_stations]
for (pi, st) in enumerate(stations_phase)
    si = findfirst(==(st), uniq_stations)
    si !== nothing && (n_st_cross[si] += 1.0)
end
mean_cc = zeros(length(uniq_stations))
for (si, st) in enumerate(uniq_stations)
    phis = findall(==(st), stations_phase)
    isempty(phis) || (mean_cc[si] = mean(cross_corr[phis]))
end

per_station_summary = Dict{String, Any}(
    "station_id" => uniq_stations,
    "n_phases" => Int32.(n_st_cross),
    "mean_cross_correlation" => mean_cc,
    "misfit_total" => zeros(length(uniq_stations)),
)

# === 7. summary ===
summary = Dict{String, Any}(
    "total_iterations" => Int32(iter_n + 1),
    "total_trials" => Int32(n_trials),
    "convergence_reason" => "single iteration (refinement pending)",
)

# === 8. 写 output.h5 ===
IO.write_output(out_path, solution, uncertainty, per_phase, per_station_summary, summary)
@info "wrote $out_path"
