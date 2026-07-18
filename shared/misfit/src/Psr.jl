# PSR misfit plugin (template)
#
# Included inside Config.{name} (dynamically created inner module).
# Template for P/S amplitude ratio - instantiated per phase pair via
# Config.use_misfit!(:Psr, operator = Misfit.Psr, ...).
#
# Config stubs (user must override):
#   pre_P()/post_P() - P window period counts
#   pre_S()/post_S() - S window period counts
# GF + obs preprocessed by Layer 0 (demean/detrend/taper/bandpass, freq-dependent).

const PSR_VALUE = :psr_value

# ── Operator 元数据 ──
outputs() = [PSR_VALUE]

export pre_P, post_P, pre_S, post_S, preprocess, process, outputs, is_freq_dependent, PSR_VALUE

is_freq_dependent() = true
# -- Config namespace (user must override) --

function pre_P()::Float64
    error("Psr.pre_P(): not implemented - return period count (e.g. 1.0)")
end

function post_P()::Float64
    error("Psr.post_P(): not implemented - return period count (e.g. 3.0)")
end

function pre_S()::Float64
    error("Psr.pre_S(): not implemented - return period count (e.g. 1.0)")
end

function post_S()::Float64
    error("Psr.post_S(): not implemented - return period count (e.g. 5.0)")
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))
const _IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

"""
    preprocess(gf_P_full, gf_S_full, obs_P_full, obs_S_full, dt,
               arrival_P, arrival_S, pre_P, post_P, pre_S, post_S, band_high)
               -> (amp_P, amp_S, obs_psr)

P/S amplitude ratio reductions.
- amp_P = GF_P[win]' * GF_P[win]  (6x6)
- amp_S = GF_S[win]' * GF_S[win]  (6x6)
- obs_psr = log10(rms(obs_P_win) / rms(obs_S_win))
Window lengths scale with band: pre_P/band_high etc. (period counts).
"""
function preprocess(
    gf_P_full::Matrix{Float64},
    gf_S_full::Matrix{Float64},
    obs_P_full::Vector{Float64},
    obs_S_full::Vector{Float64},
    dt::Float64,
    arrival_P::Int,
    arrival_S::Int,
    pre_P_p::Float64,
    post_P_p::Float64,
    pre_S_p::Float64,
    post_S_p::Float64,
    band_high::Float64,
)
    p_pre_n = max(1, round(Int, pre_P_p / band_high / dt))
    p_post_n = max(1, round(Int, post_P_p / band_high / dt))
    s_pre_n = max(1, round(Int, pre_S_p / band_high / dt))
    s_post_n = max(1, round(Int, post_S_p / band_high / dt))
    p_s = max(1, arrival_P - p_pre_n)
    p_e = min(length(obs_P_full), arrival_P + p_post_n)
    s_s = max(1, arrival_S - s_pre_n)
    s_e = min(length(obs_S_full), arrival_S + s_post_n)
    amp_P = gf_P_full[p_s:p_e, :]' * gf_P_full[p_s:p_e, :]
    amp_S = gf_S_full[s_s:s_e, :]' * gf_S_full[s_s:s_e, :]
    rms_P = _Signal.rms_amplitude(obs_P_full[p_s:p_e])
    rms_S = _Signal.rms_amplitude(obs_S_full[s_s:s_e])
    obs_psr = rms_S > 0.0 ? log10(rms_P / rms_S) : 0.0
    return amp_P, amp_S, obs_psr
end

"""
    process(phases_P, phases_S, stations, picks, station_to_idx,
            prepro_obs, prepro_gf, depths, band_high, freq_idx, pf)

Batch preprocess PSR. Pair P/S per station (same channel).
Returns a Dict mirroring the HDF5 schema:
  "channel_id"  => String[N_entries]
  "station_idx" => Int32[N_entries]
  "obs"    => Dict(freq_idx => Dict("obs_psr" => Float64[N]))
  "amp_P"  => Dict(depth => Dict(freq_idx => Float64[N, 6, 6]))
  "amp_S"  => Dict(depth => Dict(freq_idx => Float64[N, 6, 6]))
"""
function process(
    phases_P::Vector{Tuple{String, Int}},
    phases_S::Vector{Tuple{String, Int}},
    stations::Vector{_IO.StationInfo},
    picks::Vector{_IO.PhasePick},
    station_to_idx::Dict{String, Int},
    prepro_obs::Dict{String, Vector{Float64}},
    prepro_gf::Dict,  # Dict{Float64, Dict{String, Matrix{Float64}}}
    depths::Vector{Float64},
    band_high::Float64,
    freq_idx::Int,
    pf::Dict{String, Symbol},
)
    pre_P_p = pre_P()
    post_P_p = post_P()
    pre_S_p = pre_S()
    post_S_p = post_S()

    p_by_sta = Dict{Int, String}(si => pid for (pid, si) in phases_P)
    s_by_sta = Dict{Int, String}(si => pid for (pid, si) in phases_S)
    common = sort(collect(intersect(keys(p_by_sta), keys(s_by_sta))))

    obs_psr_vec = Float64[]
    amp_P_lists =
        Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    amp_S_lists =
        Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    ch_vec = String[]
    sta_vec = Int32[]

    for si in common
        s = stations[si]
        dt = s.dt
        pick = picks[station_to_idx[s.id]]
        ch_id = "$(s.network).$(s.station).$(s.channel)"
        obs_full = get(prepro_obs, ch_id, nothing)
        obs_full === nothing && continue
        n_samples = length(obs_full)

        begin_unix = _IO.parse_time_iso(s.begin_time)
        pick_P = _IO.parse_time_iso(getfield(pick, pf["P"]))
        pick_S = _IO.parse_time_iso(getfield(pick, pf["S"]))
        arrival_P = if isnan(begin_unix) || isnan(pick_P)
            n_samples ÷ 2
        else
            clamp(round(Int, (pick_P - begin_unix) / dt) + 1, 1, n_samples)
        end
        arrival_S = if isnan(begin_unix) || isnan(pick_S)
            n_samples ÷ 2
        else
            clamp(round(Int, (pick_S - begin_unix) / dt) + 1, 1, n_samples)
        end

        gf_per_depth = Dict{Float64, Matrix{Float64}}()
        all_ok = true
        for d in depths
            gf = get(prepro_gf[d], ch_id, nothing)
            if gf === nothing
                all_ok = false
                break
            end
            gf_per_depth[d] = gf
        end
        !all_ok && continue

        amp_P0, amp_S0, obs_psr = preprocess(
            gf_per_depth[depths[1]],
            gf_per_depth[depths[1]],
            obs_full,
            obs_full,
            dt,
            arrival_P,
            arrival_S,
            pre_P_p,
            post_P_p,
            pre_S_p,
            post_S_p,
            band_high,
        )
        push!(obs_psr_vec, obs_psr)
        push!(ch_vec, ch_id)
        push!(sta_vec, Int32(si))
        push!(amp_P_lists[depths[1]], amp_P0)
        push!(amp_S_lists[depths[1]], amp_S0)

        for d in depths[2:end]
            amp_P_d, amp_S_d, _ = preprocess(
                gf_per_depth[d],
                gf_per_depth[d],
                obs_full,
                obs_full,
                dt,
                arrival_P,
                arrival_S,
                pre_P_p,
                post_P_p,
                pre_S_p,
                post_S_p,
                band_high,
            )
            push!(amp_P_lists[d], amp_P_d)
            push!(amp_S_lists[d], amp_S_d)
        end
    end

    n = length(ch_vec)
    if n == 0
        return Dict(
            "channel_id" => String[],
            "station_idx" => Int32[],
            "obs" => Dict(freq_idx => Dict("obs_psr" => Float64[])),
            "amp_P" => Dict(d => Dict(freq_idx => zeros(Float64, 0, 6, 6)) for d in depths),
            "amp_S" => Dict(d => Dict(freq_idx => zeros(Float64, 0, 6, 6)) for d in depths),
        )
    end

    amp_P_arr = Dict{Float64, Array{Float64, 3}}()
    amp_S_arr = Dict{Float64, Array{Float64, 3}}()
    for d in depths
        amp_P_arr[d] = zeros(Float64, n, 6, 6)
        amp_S_arr[d] = zeros(Float64, n, 6, 6)
    end

    for i in 1:n
        for d in depths
            amp_P_arr[d][i, :, :] = amp_P_lists[d][i]
            amp_S_arr[d][i, :, :] = amp_S_lists[d][i]
        end
    end

    return Dict(
        "channel_id" => ch_vec,
        "station_idx" => sta_vec,
        "obs" => Dict(freq_idx => Dict("obs_psr" => obs_psr_vec)),
        "amp_P" => Dict(d => Dict(freq_idx => amp_P_arr[d]) for d in depths),
        "amp_S" => Dict(d => Dict(freq_idx => amp_S_arr[d]) for d in depths),
    )
end
