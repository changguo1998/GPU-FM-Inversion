# Polarity misfit plugin
# Config function stubs + preprocessing.
# GF cleaned by Layer 0 (demean/detrend/taper, no bandpass — not freq-dependent).

# ── 输出字段常量（IDE 可补全，注册时校验）──
const SYN_SIGN = :syn_sign
const DOT_VALUE = :dot_value

# ── Operator 元数据 ──
outputs() = [SYN_SIGN, DOT_VALUE]

export source_duration, preprocess, process, is_freq_dependent, outputs

is_freq_dependent() = false
# -- Config namespace (user must override) --

function source_duration()::Float64
    error("Polarity.source_duration(): not implemented - return Float64 seconds (e.g. 2.0)")
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))
const _IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

"""
    preprocess(gf_full, dt, arrival_sample, source_duration) -> gf_pol

Trim GF to polarity window [arrival, arrival+source_duration].
"""
function preprocess(
    gf_full::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    source_duration::Float64,
)
    return _Signal.trim_to_polarity_window!(gf_full, dt, arrival_sample, source_duration)
end

"""
    process(phases_pt, ptype, stations, picks, station_to_idx,
            prepro_gf, depths, pf, pol_f)

Batch polarity preprocessing for one phase type; obs_pol from manual picks
(±1/NaN), no preprocessing. Returns a Dict mirroring the HDF5 schema
(band 1): "channel_id", "station_idx", "obs", "gf".
"""
function process(
    phases_pt::Vector{Tuple{String, Int}},
    ptype::String,
    stations::Vector{_IO.StationInfo},
    picks::Vector{_IO.PhasePick},
    station_to_idx::Dict{String, Int},
    prepro_gf::Dict,  # Dict{Float64, Dict{String, Matrix{Float64}}}
    depths::Vector{Float64},
    pf::Dict{String, Symbol},
    pol_f::Dict{String, Symbol},
)
    t_source = source_duration()
    pol_field = get(pol_f, ptype, nothing)
    pol_field === nothing && return Dict(
        "channel_id" => String[],
        "station_idx" => Int32[],
        "obs" => Dict(1 => Dict("obs" => Float64[])),
        "gf" => Dict(d => Dict(1 => Dict("gf" => zeros(Float64, 0, 6, 0))) for d in depths),
    )

    # Pass 1: collect valid entries, determine common n_pol
    obs_vec = Float64[]
    ch_vec = String[]
    sta_vec = Int32[]
    gf_lists =
        Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    n_pol_list = Int[]

    for (pid, si) in phases_pt
        s = stations[si]
        dt = s.dt
        pick = picks[station_to_idx[s.id]]
        ch_id = "$(s.network).$(s.station).$(s.channel)"

        # n_samples from GF (obs waveform not needed for polarity)
        gf_first = get(prepro_gf[depths[1]], ch_id, nothing)
        gf_first === nothing && continue
        n_samples = size(gf_first, 1)

        begin_unix = _IO.parse_time_iso(s.begin_time)
        pick_time = _IO.parse_time_iso(getfield(pick, pf[ptype]))
        arrival_sample = if isnan(begin_unix) || isnan(pick_time)
            n_samples ÷ 2
        else
            clamp(round(Int, (pick_time - begin_unix) / dt) + 1, 1, n_samples)
        end

        gf_per_depth = Dict{Float64, Matrix{Float64}}()
        all_gf_ok = true
        for depth_val in depths
            gf_full = get(prepro_gf[depth_val], ch_id, nothing)
            if gf_full === nothing
                all_gf_ok = false
                break
            end
            gf_per_depth[depth_val] = gf_full
        end
        !all_gf_ok && continue

        obs_pol_int8 = getfield(pick, pol_field)::Int8
        obs_pol_val = if obs_pol_int8 == Int8(-128)
            NaN
        else
            Float64(obs_pol_int8)
        end
        push!(obs_vec, obs_pol_val)
        push!(ch_vec, ch_id)
        push!(sta_vec, Int32(si))

        # Preprocess GF at first depth to determine n_pol
        gf_pol0 = preprocess(gf_per_depth[depths[1]], dt, arrival_sample, t_source)
        n_pol = size(gf_pol0, 1)
        push!(n_pol_list, n_pol)
        push!(gf_lists[depths[1]], gf_pol0)

        for depth_val in depths[2:end]
            gf_pol = preprocess(gf_per_depth[depth_val], dt, arrival_sample, t_source)
            push!(gf_lists[depth_val], gf_pol)
        end
    end

    n_entries = length(ch_vec)
    n_pol_common = n_entries == 0 ? 0 : minimum(n_pol_list)

    if n_entries == 0 || n_pol_common == 0
        return Dict(
            "channel_id" => String[],
            "station_idx" => Int32[],
            "obs" => Dict(1 => Dict("obs" => Float64[])),
            "gf" => Dict(d => Dict(1 => Dict("gf" => zeros(Float64, 0, 6, 0))) for d in depths),
        )
    end

    # Pass 2: pre-allocate and fill
    obs_arr = zeros(Float64, n_entries)
    gf_arr = Dict{Float64, Array{Float64, 3}}()
    for d in depths
        gf_arr[d] = zeros(Float64, n_entries, 6, n_pol_common)
    end

    for i in 1:n_entries
        obs_arr[i] = obs_vec[i]
        for d in depths
            gf_trimmed = gf_lists[d][i][1:n_pol_common, :]
            gf_arr[d][i, :, :] = gf_trimmed'
        end
    end

    return Dict(
        "channel_id" => ch_vec,
        "station_idx" => sta_vec,
        "obs" => Dict(1 => Dict("obs" => obs_arr)),
        "gf" => Dict(d => Dict(1 => Dict("gf" => gf_arr[d])) for d in depths),
    )
end
