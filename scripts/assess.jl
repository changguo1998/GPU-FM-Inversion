#!/usr/bin/env julia
#
# assess.jl - 读 intermediates -> extract -> compose -> 写 misfits
#
# Usage: julia scripts/assess.jl <database.h5> <status_N.h5>
#
# Level 1 (base): EXTRACTORS[(operator, output)] transforms raw intermediates
#                 into misfit matrices.
# Level 2 (composed): COMPOSERS[operator] aggregates base misfit matrices
#                      (topologically ordered by bases).

using HDF5

using IO, Config, Aggregate

db_path = ARGS[1]
status_path = ARGS[2]

# === 1. 读 /config 元数据 ===
cfg = IO.read_config(db_path)
modules = cfg["misfit_modules"]

# === 2. 读对齐上下文 ===
stations = IO.read_stations(db_path)
N_stations = length(stations)
trials = IO.read_trials(status_path)
N_trials = length(trials.strike)

# station_idx per module (from database.h5 /{Module}/station_idx, 1-based Int32)
module_station_idx = Dict{String, Vector{Int32}}()
for m_name in modules
    grp = "/" * m_name * "/station_idx"
    h5open(
        f -> haskey(f, grp) ? (module_station_idx[m_name] = Int32.(read(f[grp]))) : nothing,
        db_path,
        "r",
    )
end

# helper: read an intermediate group into a Dict, transposed to [entries × trials]
function read_intermediate(status_path, key)
    grp = "/intermediates/$key"
    h5open(status_path, "r") do f
        !haskey(f, grp) && return Dict{String, Any}()
        d = Dict{String, Any}()
        for k in keys(f[grp])
            a = read(f[grp][k])
            # C++ writes [entries × trials] C-order; Julia reads (trials, entries).
            # Transpose to [entries × trials] for consistent indexing.
            if isa(a, AbstractMatrix) && size(a, 1) == N_trials && size(a, 2) != N_trials
                d[k] = permutedims(a)
            else
                d[k] = a
            end
        end
        return d
    end
end

# === 3. Level 1: extract ===
misfits = Dict{String, Matrix{Float64}}()
level2 = String[]
for m_name in modules
    mcfg = cfg[m_name]
    is_composed = Int8(mcfg["is_composed"]) == 1
    if is_composed
        push!(level2, m_name)
        continue
    end
    op = Symbol(mcfg["operator"])
    out = Symbol(mcfg["output"])
    phase = mcfg["phase"]
    ch = mcfg["channel"]
    key = string(op, phase)
    ch != "" && (key = key * "_" * ch)
    inter = read_intermediate(status_path, key)
    isempty(inter) && continue
    # build extractor context
    ctx = if op == :Xcorr
        (dt = stations[1].dt,)
    elseif op == :Polarity
        # obs_pol aligned to intermediate station rows. C++ indexes station_idx
        # 1-based into a 0-based array, so intermediate row j (1-based) -> station_idx (j-1);
        # row 1 is a dummy. Size obs_pol to the intermediate's first dim.
        N_sta = haskey(inter, "syn_sign") ? size(inter["syn_sign"], 1) : length(stations)
        pol_of_station = try
            picks = IO.read_phase_picks(db_path)
            sta_to_pick = Dict(p.station_id => p for p in picks)
            Int8[
                get(sta_to_pick, s.id, IO.PhasePick("", "", "", Int8(-128))).P_polarity for
                s in stations
            ]
        catch
            fill(Int8(-128), length(stations))
        end
        obs_pol = Int8[
            (j - 1 >= 1 && j - 1 <= length(pol_of_station)) ? pol_of_station[j - 1] : Int8(-128) for j in 1:N_sta
        ]
        (obs_pol = obs_pol,)
    else
        (;)
    end
    misfits[m_name] = EXTRACTORS[(op, out)](inter, ctx)
end

# === 4. Level 2: compose (topological: bases must be computed first) ===
remaining = copy(level2)
while !isempty(remaining)
    progressed = false
    for m_name in collect(remaining)
        mcfg = cfg[m_name]
        bs = String.(mcfg["bases"])
        if all(b in keys(misfits) for b in bs)
            op = Symbol(mcfg["operator"])
            out = Symbol(mcfg["output"])
            base_misfits = [misfits[b] for b in bs]
            base_station_idx = [get(module_station_idx, b, Int32[]) for b in bs]
            ctx = (N_stations = N_stations,)
            res = COMPOSERS[op](base_misfits, base_station_idx, ctx)
            misfits[m_name] = res[out]
            filter!(!=(m_name), remaining)
            progressed = true
        end
    end
    progressed || error("assess: circular or unresolved bases in $remaining")
end

# === 5. 写 /misfits/ ===
h5open(status_path, "r+") do f
    !haskey(f, "misfits") && create_group(f, "misfits")
    for (m_name, m) in misfits
        haskey(f["misfits"], m_name) && delete_object(f["misfits"], m_name)
        f["misfits"][m_name] = m
    end
end

@info "assess: wrote $(length(misfits)) misfit matrices to $status_path"
for (m_name, m) in sort(collect(misfits), by = first)
    @info "  $m_name : $(size(m))"
end

# 收敛决策: 写 $DATA_DIR/.decision.txt (driver 读取)。
# 空文件 = 收敛 → driver 停止循环。权重聚合/网格细化落地后,
# 在此用实际收敛判据决定写空或写 "continue"。
if haskey(ENV, "DATA_DIR")
    decision_path = joinpath(ENV["DATA_DIR"], ".decision.txt")
    write(decision_path, "")
    @info "assess: converged — wrote empty decision for driver"
end
