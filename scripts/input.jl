#!/usr/bin/env julia
#
# input.jl — 数据接入与初始化阶段
# 管道启动后执行一次, 完成以下工作:
#   1. 加载用户配置 (config.jl)
#   2. 通过 Config.load_*() 读外部数据 (事件/台站/波形/格林函数)
#   3. 预处理波形 (带通滤波 + 裁窗)
#   4. 写入 database.h5 (/event, /station, /channel, /gf, /{ModuleName})
#   5. 写入 status_0.h5 (初始策略, 无 trial)
#
# Usage:
#   julia scripts/input.jl <config.jl>

# === 1. 导入依赖 ===
using HDF5
using LinearAlgebra
using Dates
using Random

using StageLog

using IO, Signal, Config, Grid

# === 2. 命令行参数 & 日志 ===
config_jl = ARGS[1]
data_dir = dirname(abspath(config_jl))

StageLog.setup_logger!("input", joinpath(data_dir, "input.log"))

@info "="^70
@info "input stage started"
@info "  config   = $config_jl"
@info "  data dir = $data_dir"

# === 3. 加载用户配置 (config.jl) ===
# include config.jl, 用户在此注册插件和定义数据接口
include(abspath(config_jl))

# 读取配置中的管道参数
misfit_modules = Config.misfit_modules()
freq_bands = Config.freq_bands()
depths = Config.depths()

n_bands = length(freq_bands)
n_depths = length(depths)

# 从频带边界构造频率数组
freq_vals = sort(unique(Float64[v for (low, high) in freq_bands for v in (low, high)]))

@info "Config loaded"
@info "  misfit_modules = $misfit_modules"
@info "  freq_bands     = $freq_bands"
@info "  depths         = $depths"

# === 4. 读取外部数据 (via Config.load_*()) ===
@info "Reading external data via Config.load_*() ..."

event = Config.load_event()          # 事件信息: 经纬度/深度/震级/发震时刻
picks = Config.load_phase_picks()    # P/S 到时 + P 波极性
stations = Config.load_stations()    # 台站元数据

station_to_idx = Dict(pick.station_id => i for (i, pick) in enumerate(picks))

n_stations = length(stations)
n_picks = length(picks)

# === 5. 构建相位列表 ===
# 震相字段映射由 config.jl 定义 (Config.phase_fields/polarity_fields)
pf = Config.phase_fields()
pol_f = Config.polarity_fields()

ch_map = Dict("N" => 1, "E" => 2, "Z" => 3)  # GF 通道顺序 [N, E, D]


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

# 从震相数据提取唯一震相类型
phase_types = sort(unique([pt for (_, pt, _) in phase_list]))


@info "  event    = (lon=$(event.longitude), lat=$(event.latitude), depth=$(event.depth), M=$(event.magnitude))"
@info "  stations = $n_stations"
@info "  phases   = $n_phases"

# === 6. 构建 /station (物理台站, 去重) ===

# 按 station.id 去重, 保持原始顺序

# 按 station.id 去重, 保持原始顺序
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

# 全量 station_idx (18) → 物理 station_idx (6) 映射
full_to_phys = Int32[phys_id_to_idx[s.id] for s in stations]

# /station 表 (N_phys_stations=6 行)
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

# === 7. 加载原始波形 → /channel ===
# 原始波形, 仅验证/调试用
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

# === 8. 加载格林函数 → /gf/{depth}/{channel_id} ===
# Config.load_gf() 读 GF, 取第 3 维对应分量
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

# === 9. 预处理波形 (核心) ===
# 逐模块/频带/相位: 带通滤波 + 裁窗 + GF 预处理
@info "Preprocessing waveforms ..."

# 构建统一模块注册表
module_instances = Dict{String, Module}()
for m_name in misfit_modules
    mod = getfield(Config, Symbol(m_name))
    module_instances[m_name] = mod
end

# 预处理结果暂存, 最终组装 ModuleData
module_data = Dict{String, IO.ModuleData}()
module_results = Dict{String, Dict}()

for ptype in phase_types
    phases_pt = [(pid, si) for (pid, pt, si) in phase_list if pt == ptype]
    isempty(phases_pt) && continue

    for (m_name, mod) in module_instances
        mod_pt = Config.phase_type(Symbol(m_name))
        mod_pt != ptype && continue

        if mod.is_freq_dependent()
            band_low = mod.band_low()
            band_high = mod.band_high()
            for local_idx in 1:length(band_low)
                li = band_low[local_idx]
                hi = band_high[local_idx]
                low_cut = freq_vals[li]
                high_cut = freq_vals[hi]

                result = mod.process(
                    phases_pt,
                    ptype,
                    stations,
                    picks,
                    station_to_idx,
                    channel_data,
                    gf_data,
                    depths,
                    low_cut,
                    high_cut,
                    local_idx,
                    pf,
                )

                isempty(result["channel_id"]) && continue

                if !haskey(module_results, m_name)
                    module_results[m_name] = result
                else
                    # Merge multi-band results
                    for (band_key, band_data) in result["obs"]
                        module_results[m_name]["obs"][band_key] = band_data
                    end
                    for (d, bands) in result["gf"]
                        if !haskey(module_results[m_name]["gf"], d)
                            module_results[m_name]["gf"][d] = Dict{Int, Any}()
                        end
                        for (band_key, band_data) in bands
                            module_results[m_name]["gf"][d][band_key] = band_data
                        end
                    end
                end
            end
        else
            result = mod.process(
                phases_pt,
                ptype,
                stations,
                picks,
                station_to_idx,
                channel_data,
                gf_data,
                depths,
                pf,
                pol_f,
            )

            isempty(result["channel_id"]) && continue
            module_results[m_name] = result
        end
    end
end

@info "  preprocessing complete ($(length(misfit_modules)) modules, $(length(freq_vals)) discrete frequencies)"

# === 10. 构建 /event, /paraspace, /config ===
# /paraspace: 浮点数组; /config: 元数据 (无索引)
event_dict = Dict{String, Any}(
    "longitude" => event.longitude,
    "latitude" => event.latitude,
    "depth" => event.depth,
    "magnitude" => event.magnitude,
    "origintime" => event.origintime,
)

strike_vals = Grid.expand_axis(0.0, 5.0, Int32(71))
dip_vals = Grid.expand_axis(0.0, 5.0, Int32(19))
rake_vals = Grid.expand_axis(-90.0, 5.0, Int32(37))

paraspace = Dict{String, Any}(
    "strike" => strike_vals,
    "dip" => dip_vals,
    "rake" => rake_vals,
    "depth" => Float64.(depths),
    "frequency" => freq_vals,
)

db_config = Dict{String, Any}("misfit_modules" => misfit_modules)

# 写入 /config/{ModuleName}/
for (m_name, mod) in module_instances
    cfg_entry = Dict{String, Any}("trim" => Float64.(mod.trim()))
    if isdefined(mod, :maxlag_factor)
        cfg_entry["maxlag_factor"] = Float64(mod.maxlag_factor())
        cfg_entry["filter_order"] = Int32(mod.filter_order())
        cfg_entry["select_threshold"] = Float64(mod.select_threshold())
        cfg_entry["deselect_threshold"] = Float64(mod.deselect_threshold())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    db_config[m_name] = cfg_entry
end

# Dict → IO.ModuleData 转换
function result_to_moduledata(result::Dict)::IO.ModuleData
    obs_str = Dict{String, Matrix{Float64}}()
    obs_n2_str = Dict{String, Vector{Float64}}()
    for (bk, bv) in result["obs"]
        k = string(bk)
        obs_val = bv["obs"]
        obs_str[k] = obs_val isa Vector ? reshape(obs_val, length(obs_val), 1) : obs_val
        if haskey(bv, "obs_norm2")
            obs_n2_str[k] = bv["obs_norm2"]
        end
    end

    gf_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    syn_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    for (d, bands) in result["gf"]
        gf_str[d] = Dict{String, Array{Float64, 3}}()
        syn_str[d] = Dict{String, Array{Float64, 3}}()
        for (bk, bv) in bands
            k = string(bk)
            gf_str[d][k] = bv["gf"]
            if haskey(bv, "synamp")
                syn_str[d][k] = bv["synamp"]
            end
        end
    end

    return IO.ModuleData(
        obs = obs_str,
        obs_norm2 = obs_n2_str,
        gf = gf_str,
        synamp = syn_str,
        channel_id = result["channel_id"],
        station_idx = full_to_phys[result["station_idx"]],
    )
end
# === 11. 写入 database.h5 ===
@info "Writing database.h5 ..."
db_path = joinpath(data_dir, "database.h5")
# 组装 ModuleData → /{ModuleName}/obs + /gf
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

# === 12. 写入 status_0.h5 (初始策略) ===
# 记录 /strategy: depth_indices + 频带索引
@info "Writing status_0.h5 ..."

strategy = IO.Strategy(Int32.(1:n_depths), Int32.(1:n_bands), Int32(0))

status0_path = joinpath(data_dir, "status_0.h5")
h5open(status0_path, "w") do f
end
IO.write_strategy(status0_path, strategy)
@info "  $status0_path written"

# === 13. 输出摘要 ===
@info ""
@info "Stage complete:"
@info "  $(basename(db_path)) : /station ($n_phys_stations rows)"
@info "  $(basename(db_path)) : /channel ($(length(channel_data)) channels)"
@info "  $(basename(db_path)) : /gf ($(length(gf_data)) depths)"
mod_names_str = join(sort(collect(keys(module_data))), ", ")
@info "  $(basename(db_path)) : /$mod_names_str (modules)"
@info "  $(basename(db_path)) : /config, /event"
@info "  $(basename(status0_path)) : /strategy (initial grid, no trials)"
@info "  Phys stations: $n_phys_stations | Channels: $n_stations | Phases: $n_phases | Depths: $n_depths | Freq vals: $(length(freq_vals))"
@info ""
@info "="^70
