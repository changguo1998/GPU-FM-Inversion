# Polarity misfit plugin
#
# Included inside Config.Polarity (dynamically created inner module).
# Config function stubs + preprocessing logic.

export trim, preprocess, process, is_freq_dependent

is_freq_dependent() = false
# -- Config namespace (user must override) --

function trim()::Vector{Float64}
    error("Polarity.trim(): not implemented — return [t_pre, t_post]  (e.g. [0.0, 2.0])")
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))
const _IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

"""
    preprocess(gf, dt, arrival_sample, t_source, obs_polarity) -> (gf_pol, obs_pol)

Trim GF to polarity window.
"""
function preprocess(
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    t_source::Float64,
    obs_polarity::Int8,
)
    gf_pol = _Signal.trim_to_polarity_window!(gf, dt, arrival_sample, t_source)
    obs_pol_float = if obs_polarity == Int8(-128)
        NaN
    else
        Float64(obs_polarity)
    end
    return gf_pol, obs_pol_float
end

"""
    process(phases_pt, ptype, stations, picks, station_to_idx, channel_data,
            gf_data, depths, pf, pol_f)

Batch preprocess Polarity for one phase type.
Returns a Dict mirroring the HDF5 schema (band 1, no freq filtering):
  "channel_id"  => String[N_entries]
  "station_idx" => Int32[N_entries]
  "obs" => Dict(1 => Dict("obs" => Float64[N_entries]))
  "gf"  => Dict(depth => Dict(1 => Dict("gf" => Float64[N_entries, 6, N_pol])))
"""
function process(
    phases_pt::Vector{Tuple{String, Int}},
    ptype::String,
    stations::Vector{_IO.StationInfo},
    picks::Vector{_IO.PhasePick},
    station_to_idx::Dict{String, Int},
    channel_data::Dict{String, Vector{Float64}},
    gf_data::Dict,  # Dict{Float64, Dict{String, Matrix{Float64}}}
    depths::Vector{Float64},
    pf::Dict{String, Symbol},
    pol_f::Dict{String, Symbol},
)
    t_source = trim()[2]
    pol_field = get(pol_f, ptype, nothing)
    pol_field === nothing && return Dict(
        "channel_id"  => String[],
        "station_idx" => Int32[],
        "obs" => Dict(1 => Dict("obs" => Float64[])),
        "gf"  => Dict(d => Dict(1 => Dict("gf" => zeros(Float64, 0, 6, 0))) for d in depths),
    )

    # Pass 1: collect valid entries, determine common n_pol
    obs_vec = Float64[]
    ch_vec = String[]
    sta_vec = Int32[]
    gf_lists = Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    n_pol_list = Int[]

    for (pid, si) in phases_pt
        s = stations[si]
        dt = s.dt
        pick = picks[station_to_idx[s.id]]
        ch_id = "$(s.network).$(s.station).$(s.channel)"
        wf = channel_data[ch_id]
        n_samples = length(wf)

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
            gf_full = get(gf_data[depth_val], ch_id, nothing)
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
        gf_pol0, _ = preprocess(gf_per_depth[depths[1]], dt, arrival_sample, t_source, obs_pol_int8)
        n_pol = size(gf_pol0, 1)
        push!(n_pol_list, n_pol)
        push!(gf_lists[depths[1]], gf_pol0)

        for depth_val in depths[2:end]
            gf_pol, _ = preprocess(gf_per_depth[depth_val], dt, arrival_sample, t_source, obs_pol_int8)
            push!(gf_lists[depth_val], gf_pol)
        end
    end

    n_entries = length(ch_vec)
    n_pol_common = n_entries == 0 ? 0 : minimum(n_pol_list)

    if n_entries == 0 || n_pol_common == 0
        return Dict(
            "channel_id"  => String[],
            "station_idx" => Int32[],
            "obs" => Dict(1 => Dict("obs" => Float64[])),
            "gf"  => Dict(d => Dict(1 => Dict("gf" => zeros(Float64, 0, 6, 0))) for d in depths),
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
        "channel_id"  => ch_vec,
        "station_idx" => sta_vec,
        "obs" => Dict(
            1 => Dict("obs" => obs_arr),
        ),
        "gf" => Dict(
            d => Dict(
                1 => Dict("gf" => gf_arr[d]),
            ) for d in depths
        ),
    )
end
