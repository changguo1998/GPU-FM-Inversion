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

using IO, Config, Aggregate, Misfit

db_path = ARGS[1]
status_path = ARGS[2]

# === 1. 读 /config 元数据 ===
cfg = IO.read_config(db_path)
modules = cfg["misfit_modules"]

# === 2. 读对齐上下文 ===
stations = IO.read_stations(db_path)
N_stations = length(stations)
trials = IO.read_trials(status_path)
N_trials = length(trials.strike_idx)

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

function read_database_dataset(path)
    return h5open(f -> read(f[path]), db_path, "r")
end

function intermediate_key(mcfg)
    key = string(mcfg["operator"], mcfg["phase"])
    mcfg["channel"] != "" && (key *= "_" * mcfg["channel"])
    return key
end

function evaluate_dsl_objective(name)
    mcfg = cfg[name]
    expr = Misfit.decode_expression(cfg["objectives"][name])
    bases = Int8(mcfg["is_composed"]) == 1 ? String.(mcfg["bases"]) : [name]
    isempty(bases) && error("assess: objective $name has no waveform bases")

    base_ids = Dict(base => String.(read_database_dataset("/$base/channel_id")) for base in bases)
    target_ids = copy(base_ids[first(bases)])
    for base in Iterators.drop(bases, 1)
        available = Set(base_ids[base])
        filter!(id -> id in available, target_ids)
    end
    isempty(target_ids) && error("assess: objective $name bases have no common channel_id")
    base_rows = Dict(
        base => begin
            row_of = Dict(id => row for (row, id) in enumerate(base_ids[base]))
            [row_of[id] for id in target_ids]
        end for base in bases
    )
    intermediate_cache =
        Dict(base => read_intermediate(status_path, intermediate_key(cfg[base])) for base in bases)
    observation_cache = Dict(base => read_database_dataset("/$base/obs/1/obs") for base in bases)

    function waveform_base(node)
        candidates = filter(bases) do base
            bcfg = cfg[base]
            bcfg["phase"] == string(node.phase) &&
                (bcfg["channel"] == "" || bcfg["channel"] == something(node.channel, ""))
        end
        length(candidates) == 1 ||
            error("assess: objective $name cannot uniquely resolve $(node.phase) waveform base")
        return only(candidates)
    end

    function repeated_column(values)
        return repeat(reshape(Float64.(values), :, 1), 1, N_trials)
    end

    function resolve_primitive(call)
        op = call.op
        node = op isa Union{Misfit.MaxCCOp, Misfit.LagCCOp} ? call.args[1] : only(call.args)
        base = waveform_base(node)
        rows = base_rows[base]
        inter = intermediate_cache[base]
        if op isa Misfit.MaxCCOp
            return Float64.(inter["cc_max"][rows, :])
        elseif op isa Misfit.LagCCOp
            return Float64.(inter["best_lag"][rows, :]) .* stations[1].dt
        end

        if node.role == :observed
            obs = observation_cache[base][rows, :]
            if op isa Misfit.EnergyOp
                return repeated_column(vec(sum(abs2, obs; dims = 2)))
            elseif op isa Misfit.RMSOp
                return repeated_column(sqrt.(vec(sum(abs2, obs; dims = 2)) ./ size(obs, 2)))
            elseif op isa Misfit.AmpScaleOp
                return repeated_column([Misfit.ampScale(view(obs, row, :)) for row in axes(obs, 1)])
            elseif op isa Misfit.SignScaleOp
                return repeated_column([
                    Misfit.signScale(view(obs, row, :)) for row in axes(obs, 1)
                ])
            end
        else
            if op isa Misfit.EnergyOp
                return Float64.(inter["syn_energy"][rows, :])
            elseif op isa Misfit.RMSOp
                n_samples = size(observation_cache[base], 2)
                return sqrt.(Float64.(inter["syn_energy"][rows, :]) ./ n_samples)
            elseif op isa Misfit.AmpScaleOp
                return Float64.(inter["amp_scale"][rows, :])
            elseif op isa Misfit.SignScaleOp
                return Float64.(inter["sign_scale"][rows, :])
            end
        end
        error("assess: objective $name has unsupported primitive $(typeof(op))")
    end

    value = Misfit.evaluate_pipeline(expr, resolve_primitive)
    result = if value isa Number
        fill(Float64(value), 1, N_trials)
    elseif value isa AbstractVector && length(value) == N_trials
        reshape(Float64.(value), 1, N_trials)
    elseif value isa AbstractMatrix && size(value, 2) == N_trials
        Float64.(value)
    else
        error("assess: objective $name produced invalid shape $(size(value))")
    end
    all(isfinite, result) || error("assess: objective $name produced non-finite values")
    return result
end

function evaluate_psr(bases)
    length(bases) == 2 || error("assess: PSR requires P and S bases")
    p_base = only(filter(base -> cfg[base]["phase"] == "P", bases))
    s_base = only(filter(base -> cfg[base]["phase"] == "S", bases))
    p_ids = String.(read_database_dataset("/$p_base/channel_id"))
    s_ids = String.(read_database_dataset("/$s_base/channel_id"))
    s_row = Dict(id => row for (row, id) in enumerate(s_ids))
    p_rows = Int[]
    s_rows = Int[]
    for (row, id) in enumerate(p_ids)
        haskey(s_row, id) || continue
        push!(p_rows, row)
        push!(s_rows, s_row[id])
    end
    isempty(p_rows) && error("assess: PSR bases have no common channel_id")

    p_inter = read_intermediate(status_path, intermediate_key(cfg[p_base]))
    s_inter = read_intermediate(status_path, intermediate_key(cfg[s_base]))
    p_obs = read_database_dataset("/$p_base/obs/1/obs")
    s_obs = read_database_dataset("/$s_base/obs/1/obs")
    p_obs_energy = vec(read_database_dataset("/$p_base/obs/1/obs_norm2"))[p_rows]
    s_obs_energy = vec(read_database_dataset("/$s_base/obs/1/obs_norm2"))[s_rows]
    p_samples = fill(size(p_obs, 2), length(p_rows))
    s_samples = fill(size(s_obs, 2), length(s_rows))
    return Aggregate.psr_residual(
        p_obs_energy,
        p_samples,
        p_inter["syn_energy"][p_rows, :],
        s_obs_energy,
        s_samples,
        s_inter["syn_energy"][s_rows, :],
    )
end

function evaluate_polarity(base)
    mcfg = cfg[base]
    mcfg["phase"] == "P" || error("assess: normalized polarity requires a P base")
    inter = read_intermediate(status_path, intermediate_key(mcfg))
    obs = read_database_dataset("/$base/obs/1/obs")
    observed_signed = [
        Misfit.ampScale(view(obs, row, :)) * Misfit.signScale(view(obs, row, :)) for
        row in axes(obs, 1)
    ]
    return Aggregate.normalized_polarity_residual(
        observed_signed,
        inter["amp_scale"],
        inter["sign_scale"],
    )
end

# === 3. Level 1: extract ===
misfits = Dict{String, Matrix{Float64}}()
level2 = String[]
dsl_names = haskey(cfg, "objectives") ? Set(String.(keys(cfg["objectives"]))) : Set{String}()
for m_name in modules
    m_name in dsl_names && continue
    mcfg = cfg[m_name]
    is_composed = Int8(mcfg["is_composed"]) == 1
    if is_composed
        push!(level2, m_name)
        continue
    end
    op = Symbol(mcfg["operator"])
    out = Symbol(mcfg["output"])
    key = intermediate_key(mcfg)
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
            if op == :Psr
                misfits[m_name] = evaluate_psr(bs)
            elseif op == :Polarity
                misfits[m_name] = evaluate_polarity(only(bs))
            else
                base_misfits = [misfits[b] for b in bs]
                base_station_idx = [get(module_station_idx, b, Int32[]) for b in bs]
                ctx = (N_stations = N_stations,)
                res = COMPOSERS[op](base_misfits, base_station_idx, ctx)
                misfits[m_name] = res[out]
            end
            filter!(!=(m_name), remaining)
            progressed = true
        end
    end
    progressed || error("assess: circular or unresolved bases in $remaining")
end


# === 4b. DSL objectives: evaluate stored expression trees from primitive intermediates ===
for m_name in modules
    m_name in dsl_names || continue
    misfits[m_name] = evaluate_dsl_objective(m_name)
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
