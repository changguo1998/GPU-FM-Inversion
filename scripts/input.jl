#!/usr/bin/env julia
#
# input.jl - 数据接入与初始化阶段
# 管道启动后执行一次, 完成以下工作:
#   1. 加载用户配置 (config.jl)
#   2. 通过 Config.load_*() 读外部数据 (事件/台站/波形/格林函数)
#   3. 预处理波形 (带通滤波 + 裁窗)
#   4. 写入 database.h5 (/event, /station, /channel, /gf, /paraspace, /config, /{ModuleName})
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

# 全量 station_idx -> 物理 station_idx 映射
full_to_phys = Int32[phys_id_to_idx[s.id] for s in stations]

# /station 表 (N_phys_stations 行)
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

# === 7. 加载原始波形 -> /channel ===
# 原始波形, 写入 /channel 并作为预处理输入
@info "Building /channel - raw waveforms ..."

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

# === 8. 加载格林函数 -> /gf/{depth}/{channel_id} ===
# Config.load_gf() 读 GF, 取第 3 维对应分量
@info "Loading Green's functions via Config.load_gf() ..."

"""加载所有 (station, depth) 的格林函数, 取指定通道分量。"""
function load_greens_functions(stations, ch_map, event, depths)
    gf_data = Dict{Float64, Dict{String, Matrix{Float64}}}()
    n_skip = 0
    n_load = 0
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
    return gf_data
end

gf_data = load_greens_functions(stations, ch_map, event, depths)

# === 9. 预处理波形 (核心) ===
# 逐模块/频带/相位: 带通滤波 + 裁窗 + GF 预处理
@info "Preprocessing waveforms ..."

# 构建统一模块注册表
module_instances = Dict{String, Module}()
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) && continue   # Level 2 无 instance module
    module_instances[m_name] = getfield(Config, sym)
end

# === 9a. Layer 0: shared full-waveform preprocessing ===
# 完整波形 demean/detrend/taper/bandpass (per band); 独立于算子
@info "Layer 0: shared preprocessing (demean/detrend/taper/bandpass) per band ..."
ch_dt = Dict{String, Float64}()
for s in stations
    ch_dt["$(s.network).$(s.station).$(s.channel)"] = s.dt
end

# collect unique (low_cut, high_cut) bands from all freq-dependent modules
all_bands = Set{Tuple{Float64, Float64}}()
for (_, mod) in module_instances
    mod.is_freq_dependent() || continue
    bl, bh = mod.band_low(), mod.band_high()
    for i in 1:length(bl)
        push!(all_bands, (freq_vals[bl[i]], freq_vals[bh[i]]))
    end
end

prepro_obs = Dict{Tuple{Float64, Float64}, Dict{String, Vector{Float64}}}()       # (lo,hi) -> ch_id -> wf
prepro_gf = Dict{Tuple{Float64, Float64}, Dict{Float64, Dict{String, Matrix{Float64}}}}()  # (lo,hi) -> depth -> ch_id -> gf
for (lo, hi) in all_bands
    po = Dict{String, Vector{Float64}}()
    for (ch_id, wf_raw) in channel_data
        po[ch_id] = Signal.preprocess_waveform!(copy(wf_raw), ch_dt[ch_id], lo, hi)
    end
    prepro_obs[(lo, hi)] = po
    pg = Dict{Float64, Dict{String, Matrix{Float64}}}()
    for d in depths
        pd = Dict{String, Matrix{Float64}}()
        for (ch_id, gf_raw) in gf_data[d]
            g = copy(gf_raw)
            for c in 1:size(g, 2)
                g[:, c] = Signal.preprocess_waveform!(g[:, c], ch_dt[ch_id], lo, hi)
            end
            pd[ch_id] = g
        end
        pg[d] = pd
    end
    prepro_gf[(lo, hi)] = pg
end

# basic-clean GF (no bandpass) for Polarity (is_freq_dependent=false)
prepro_gf_basic = Dict{Float64, Dict{String, Matrix{Float64}}}()
for d in depths
    pd = Dict{String, Matrix{Float64}}()
    for (ch_id, gf_raw) in gf_data[d]
        g = copy(gf_raw)
        for c in 1:size(g, 2)
            g[:, c] =
                Signal.preprocess_waveform!(g[:, c], ch_dt[ch_id], 0.0, 0.0; do_bandpass = false)
        end
        pd[ch_id] = g
    end
    prepro_gf_basic[d] = pd
end
@info "  Layer 0 complete: $(length(prepro_obs)) bands x $(length(channel_data)) channels"

# 预处理共享上下文 (NamedTuple 避免长参数列表)
ctx = (
    stations = stations,
    picks = picks,
    station_to_idx = station_to_idx,
    depths = depths,
    freq_vals = freq_vals,
    pf = pf,
    pol_f = pol_f,
    prepro_obs = prepro_obs,
    prepro_gf = prepro_gf,
    prepro_gf_basic = prepro_gf_basic,
)

"""合并多频带 result r 到已累积的 prev (prev 为 nothing 时返回 r 本身)。"""
function merge_band_result(prev, r)
    if prev === nothing
        return r
    end
    for (band_key, band_data) in r["obs"]
        prev["obs"][band_key] = band_data
    end
    for (d, bands) in r["gf"]
        if !haskey(prev["gf"], d)
            prev["gf"][d] = Dict{Int, Any}()
        end
        for (band_key, band_data) in bands
            prev["gf"][d][band_key] = band_data
        end
    end
    return prev
end

"""对单模块执行预处理, 返回累积 result (nothing 表示无有效数据)。

freq-dependent 模块跨频带合并; 非 freq 模块单次 process。"""
function preprocess_module(mod, phases_pt, ptype, ctx, prev)
    if mod.is_freq_dependent()
        band_low = mod.band_low()
        band_high = mod.band_high()
        result = prev
        for local_idx in 1:length(band_low)
            lo = ctx.freq_vals[band_low[local_idx]]
            hi = ctx.freq_vals[band_high[local_idx]]
            r = mod.process(
                phases_pt,
                ptype,
                ctx.stations,
                ctx.picks,
                ctx.station_to_idx,
                ctx.prepro_obs[(lo, hi)],
                ctx.prepro_gf[(lo, hi)],
                ctx.depths,
                hi,
                local_idx,
                ctx.pf,
            )
            isempty(r["channel_id"]) && continue
            result = merge_band_result(result, r)
        end
        return result
    else
        r = mod.process(
            phases_pt,
            ptype,
            ctx.stations,
            ctx.picks,
            ctx.station_to_idx,
            ctx.prepro_gf_basic,
            ctx.depths,
            ctx.pf,
            ctx.pol_f,
        )
        isempty(r["channel_id"]) && return prev
        return r
    end
end

# 预处理结果暂存, 最终组装 ModuleData
module_data = Dict{String, IO.ModuleData}()
module_results = Dict{String, Dict}()

for ptype in phase_types
    phases_pt = [(pid, si) for (pid, pt, si) in phase_list if pt == ptype]
    isempty(phases_pt) && continue
    for (m_name, mod) in module_instances
        (m_name == "Psr" || Config.phase_type(Symbol(m_name)) != ptype) && continue
        prev = get(module_results, m_name, nothing)
        result = preprocess_module(mod, phases_pt, ptype, ctx, prev)
        result !== nothing && (module_results[m_name] = result)
    end
end

# Psr special: P/S pair across ptype (not handled in per-ptype loop)
"""逐频带累积 Psr 算子结果 (P/S 配对)。返回累积 result (nothing 表示无有效数据)。"""
function preprocess_psr(
    psr_mod,
    phases_P,
    phases_S,
    stations,
    picks,
    station_to_idx,
    ctx,
    depths,
    freq_vals,
    pf,
)
    bl, bh = psr_mod.band_low(), psr_mod.band_high()
    acc = nothing
    for local_idx in 1:length(bl)
        lo, hi = freq_vals[bl[local_idx]], freq_vals[bh[local_idx]]
        r = psr_mod.process(
            phases_P,
            phases_S,
            stations,
            picks,
            station_to_idx,
            ctx.prepro_obs[(lo, hi)],
            ctx.prepro_gf[(lo, hi)],
            depths,
            hi,
            local_idx,
            pf,
        )
        isempty(r["channel_id"]) && continue
        acc = merge_band_result(acc, r)
    end
    return acc
end

psr_mod = get(module_instances, "Psr", nothing)
if psr_mod !== nothing
    phases_P = [(pid, si) for (pid, pt, si) in phase_list if pt == "P"]
    phases_S = [(pid, si) for (pid, pt, si) in phase_list if pt == "S"]
    psr_result = preprocess_psr(
        psr_mod,
        phases_P,
        phases_S,
        stations,
        picks,
        station_to_idx,
        ctx,
        depths,
        freq_vals,
        pf,
    )
    psr_result !== nothing && (module_results["Psr"] = psr_result)
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
# Level 1: 有 instance module 的实例（含预处理参数）
for (m_name, mod) in module_instances
    sym = Symbol(m_name)
    cfg_entry = Dict{String, Any}()
    if isdefined(mod, :trim)
        cfg_entry["trim"] = Float64.(mod.trim())
    end
    if isdefined(mod, :max_lag_periods)
        cfg_entry["max_lag_periods"] = Float64(mod.max_lag_periods())
        cfg_entry["filter_order"] = Int32(mod.filter_order())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    if isdefined(mod, :source_duration)
        cfg_entry["source_duration"] = Float64(mod.source_duration())
    end
    if isdefined(mod, :pre_P)
        cfg_entry["pre_P"] = Float64(mod.pre_P())
        cfg_entry["post_P"] = Float64(mod.post_P())
        cfg_entry["pre_S"] = Float64(mod.pre_S())
        cfg_entry["post_S"] = Float64(mod.post_S())
        cfg_entry["band_low"] = mod.band_low()
        cfg_entry["band_high"] = mod.band_high()
    end
    cfg_entry["operator"] = string(nameof(Config.operator_module(sym)))
    cfg_entry["output"] = string(Config.output_field(sym))
    cfg_entry["is_composed"] = Int8(0)
    cfg_entry["phase"] = Config.phase_type(sym)
    ch = Config.channel_of(sym)
    cfg_entry["channel"] = ch === nothing ? "" : ch
    db_config[m_name] = cfg_entry
end

# Level 2: composed 实例（无 instance module、无预处理参数）
for m_name in misfit_modules
    sym = Symbol(m_name)
    Config.is_composed(sym) || continue
    db_config[m_name] = Dict{String, Any}(
        "operator" => string(nameof(Config.operator_module(sym))),
        "output" => string(Config.output_field(sym)),
        "is_composed" => Int8(1),
        "bases" => String.(Config.bases_of(sym)),
    )
end

# Dict -> IO.ModuleData 转换
function result_to_moduledata(result::Dict)::IO.ModuleData
    obs_str = Dict{String, Matrix{Float64}}()
    obs_n2_str = Dict{String, Vector{Float64}}()
    obs_psr_str = Dict{String, Vector{Float64}}()
    for (bk, bv) in result["obs"]
        k = string(bk)
        if haskey(bv, "obs")
            obs_val = bv["obs"]
            obs_str[k] = obs_val isa Vector ? reshape(obs_val, length(obs_val), 1) : obs_val
            if haskey(bv, "obs_norm2")
                obs_n2_str[k] = bv["obs_norm2"]
            end
        elseif haskey(bv, "obs_psr")
            obs_psr_str[k] = bv["obs_psr"]
        end
    end

    gf_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    syn_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    if haskey(result, "gf")
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
    end

    # Xcorr per-lag reductions
    synamp_lag_str = Dict{Float64, Dict{String, Array{Float64, 4}}}()
    if haskey(result, "synamp_lag")
        for (d, bands) in result["synamp_lag"]
            synamp_lag_str[d] = Dict{String, Array{Float64, 4}}()
            for (bk, bv) in bands
                synamp_lag_str[d][string(bk)] = bv
            end
        end
    end
    dog_lag_str = Dict{String, Array{Float64, 3}}()
    if haskey(result, "dot_obs_gf_lag")
        for (bk, bv) in result["dot_obs_gf_lag"]
            dog_lag_str[string(bk)] = bv
        end
    end

    # PSR reductions
    amp_P_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    amp_S_str = Dict{Float64, Dict{String, Array{Float64, 3}}}()
    if haskey(result, "amp_P")
        for (d, bands) in result["amp_P"]
            amp_P_str[d] = Dict{String, Array{Float64, 3}}()
            for (bk, bv) in bands
                amp_P_str[d][string(bk)] = bv
            end
        end
    end
    if haskey(result, "amp_S")
        for (d, bands) in result["amp_S"]
            amp_S_str[d] = Dict{String, Array{Float64, 3}}()
            for (bk, bv) in bands
                amp_S_str[d][string(bk)] = bv
            end
        end
    end

    return IO.ModuleData(
        obs = obs_str,
        obs_norm2 = obs_n2_str,
        gf = gf_str,
        synamp = syn_str,
        synamp_lag = synamp_lag_str,
        dot_obs_gf_lag = dog_lag_str,
        amp_P = amp_P_str,
        amp_S = amp_S_str,
        obs_psr = obs_psr_str,
        channel_id = result["channel_id"],
        station_idx = full_to_phys[result["station_idx"]],
    )
end
# === 11. 写入 database.h5 ===
@info "Writing database.h5 ..."
db_path = joinpath(data_dir, "database.h5")
# 组装 ModuleData -> /{ModuleName}/obs + /gf
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

# persist Layer 0 intermediate (debug): /preprocess, /gf_preprocessed
h5open(db_path, "r+") do f
    pp = create_group(f, "preprocess")
    for ((lo, hi), chs) in prepro_obs
        bkey = "$(lo)_$(hi)"
        cg = create_group(pp, bkey)
        for (ch_id, wf) in chs
            cg[ch_id] = wf
        end
    end
    gp = create_group(f, "gf_preprocessed")
    for ((lo, hi), depths_dict) in prepro_gf
        bkey = "$(lo)_$(hi)"
        bg = create_group(gp, bkey)
        for (d, chs) in depths_dict
            dg = create_group(bg, string(d))
            for (ch_id, gf) in chs
                dg[ch_id] = gf
            end
        end
    end
end
mod_summary = join(
    [
        "$(length(module_results[mn]["channel_id"])) $mn" for
        mn in sort(collect(keys(module_results)))
    ],
    ", ",
)
@info "  phase metadata written ($mod_summary)"

# === 12. 写入 status_0.h5 (初始策略) ===
# /strategy: 全空间 5° 默认网格 (SDR) + 全 depth/freq 索引, iteration 0
@info "Writing status_0.h5 ..."

g0 = Grid.default_grid()
strategy = IO.Strategy(
    g0.strike0,
    g0.dstrike,
    g0.nstrike,
    g0.dip0,
    g0.ddip,
    g0.ndip,
    g0.rake0,
    g0.drake,
    g0.nrake,
    Int32.(1:n_depths),
    Int32.(1:n_bands),
    Int32(0),
)

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
