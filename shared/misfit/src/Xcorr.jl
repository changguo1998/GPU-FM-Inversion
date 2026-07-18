# XCorr misfit plugin (template)
#
# Included inside Config.{name} (dynamically created inner module).
# Template for cross-correlation misfit - instantiated per phase via
# `Config.use_misfit!(:XcorrP, from = :Xcorr)`.
#
# Config stubs (user must override):
#   trim()            - time window [pre, post] period counts relative to arrival
#   max_lag_periods() - max cross-correlation lag in period counts
#   filter_order()    - Butterworth filter order (Layer 0 bandpass)
#   band_low()/band_high() - freq-band indices into /paraspace/frequency

# ── 输出字段常量（IDE 可补全，注册时校验）──
const CC_MAX = :cc_max
const BEST_LAG = :best_lag

# ── Operator 元数据 ──
outputs() = [CC_MAX, BEST_LAG]

export trim, max_lag_periods, filter_order, outputs
export preprocess, process, is_freq_dependent, band_low, band_high


is_freq_dependent() = true
# -- Config namespace (user must override) --

function trim()::Vector{Float64}
    error("Xcorr.trim(): not implemented - return [-pre_periods, post_periods]  (e.g. [-2.0, 5.0])")
end

function max_lag_periods()::Float64
    error("Xcorr.max_lag_periods(): not implemented - return Float64 period count (e.g. 3.0)")
end

function filter_order()::Int
    error("Xcorr.filter_order(): not implemented - return Int  (e.g. 4)")
end

function band_low()::Vector{Int32}
    error("Xcorr.band_low(): not implemented - return Vector{Int32} of freq-band low indices")
end

function band_high()::Vector{Int32}
    error("Xcorr.band_high(): not implemented - return Vector{Int32} of freq-band high indices")
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))
const _IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

"""
    preprocess(gf_full, obs_win, dt, arrival_sample, pre_periods, post_periods,
               band_high, max_lag_periods)
               -> (obs_norm2, synamp_lag, dot_obs_gf_lag)

Compute per-lag reductions for cross-correlation.
- obs_win: fixed trimmed obs window [arrival-pre, arrival+post] (already preprocessed)
- gf_full: full preprocessed GF waveform [N_full, 6]
- synamp_lag[l] = gf_full[win-l]' * gf_full[win-l]  (6x6)
- dot_obs_gf_lag[l] = obs_win' * gf_full[win-l]     (6-vector)
- obs_norm2 = obs_win' * obs_win
"""
function preprocess(
    gf_full::Matrix{Float64},
    obs_win::Vector{Float64},
    dt::Float64,
    arrival_sample::Int,
    pre_periods::Float64,
    post_periods::Float64,
    band_high::Float64,
    max_lag_periods::Float64,
)
    pre_sec = pre_periods / band_high
    post_sec = post_periods / band_high
    pre_n = max(1, round(Int, pre_sec / dt))
    post_n = max(1, round(Int, post_sec / dt))
    nt_win = pre_n + post_n + 1
    max_lag_sec = max_lag_periods / band_high
    max_lag_n = min(round(Int, max_lag_sec / dt), (nt_win - 1) ÷ 2)
    L = 2 * max_lag_n + 1

    obs_norm2 = sum(abs2, obs_win)
    synamp_lag = zeros(Float64, 6, 6, L)
    dog_lag = zeros(Float64, 6, L)

    # window in full-gf coordinates: [arrival-pre_n, arrival+post_n]
    w_start = arrival_sample - pre_n
    for (li, lag) in enumerate((-max_lag_n):max_lag_n)
        s = w_start - lag                     # syn full window start
        e = s + nt_win - 1
        if s < 1 || e > size(gf_full, 1)
            continue                          # lag out of range, leave zeros
        end
        gf_sub = gf_full[s:e, :]
        synamp_lag[:, :, li] = gf_sub' * gf_sub
        dog_lag[:, li] = gf_sub' * obs_win
    end
    return obs_norm2, synamp_lag, dog_lag
end

"""
    process(phases_pt, ptype, stations, picks, station_to_idx,
            prepro_obs, prepro_gf, depths, band_high, freq_idx, pf)

Batch preprocess XCorr for one phase type at one frequency band.
Consumes Layer 0 preprocessed waveforms (prepro_obs/prepro_gf, already
demeaned/detrended/tapered/bandpassed). Computes per-lag reductions.

Returns a Dict mirroring the HDF5 schema:
  "channel_id"      => String[N_entries]
  "station_idx"     => Int32[N_entries]
  "obs"             => Dict(freq_idx => Dict("obs" => Float64[N, nt_win],
                                             "obs_norm2" => Float64[N]))
  "synamp_lag"      => Dict(depth => Dict(freq_idx => Float64[N, 6, 6, L]))
  "dot_obs_gf_lag"  => Dict(freq_idx => Float64[N, 6, L])
"""
function process(
    phases_pt::Vector{Tuple{String, Int}},
    ptype::String,
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
    trim_win = trim()
    pre_periods = abs(trim_win[1])
    post_periods = abs(trim_win[2])
    max_lag_p = max_lag_periods()

    # Pass 1: collect valid entries, determine L
    obs_win_list = Vector{Vector{Float64}}()
    obs_n2_list = Float64[]
    dog_lag_list = Vector{Matrix{Float64}}()
    synamp_lag_lists =
        Dict{Float64, Vector{Array{Float64, 3}}}(d => Vector{Array{Float64, 3}}() for d in depths)
    ch_vec = String[]
    sta_vec = Int32[]

    for (pid, si) in phases_pt
        s = stations[si]
        dt = s.dt
        pick = picks[station_to_idx[s.id]]
        ch_id = "$(s.network).$(s.station).$(s.channel)"
        obs_full = get(prepro_obs, ch_id, nothing)
        obs_full === nothing && continue
        n_samples = length(obs_full)

        begin_unix = _IO.parse_time_iso(s.begin_time)
        pick_time = _IO.parse_time_iso(getfield(pick, pf[ptype]))
        arrival_sample = if isnan(begin_unix) || isnan(pick_time)
            n_samples ÷ 2
        else
            clamp(round(Int, (pick_time - begin_unix) / dt) + 1, 1, n_samples)
        end

        # obs fixed window
        pre_n = max(1, round(Int, pre_periods / band_high / dt))
        post_n = max(1, round(Int, post_periods / band_high / dt))
        nt_win = pre_n + post_n + 1
        start_idx = max(1, arrival_sample - pre_n)
        end_idx = min(n_samples, arrival_sample + post_n)
        if end_idx - start_idx + 1 < nt_win
            continue                      # window clamped, skip entry
        end
        obs_win = obs_full[start_idx:end_idx]

        # gf full per depth
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

        # per-lag reductions at first depth (determines L)
        obs_n2, synamp_lag0, dog_lag0 = preprocess(
            gf_per_depth[depths[1]],
            obs_win,
            dt,
            arrival_sample,
            pre_periods,
            post_periods,
            band_high,
            max_lag_p,
        )
        push!(obs_win_list, obs_win)
        push!(obs_n2_list, obs_n2)
        push!(dog_lag_list, dog_lag0)
        push!(synamp_lag_lists[depths[1]], synamp_lag0)
        push!(ch_vec, ch_id)
        push!(sta_vec, Int32(si))

        for depth_val in depths[2:end]
            _, synamp_lag_d, _ = preprocess(
                gf_per_depth[depth_val],
                obs_win,
                dt,
                arrival_sample,
                pre_periods,
                post_periods,
                band_high,
                max_lag_p,
            )
            push!(synamp_lag_lists[depth_val], synamp_lag_d)
        end
    end

    n_entries = length(ch_vec)
    if n_entries == 0
        return Dict(
            "channel_id" => String[],
            "station_idx" => Int32[],
            "obs" =>
                Dict(freq_idx => Dict("obs" => zeros(Float64, 0, 0), "obs_norm2" => Float64[])),
            "synamp_lag" => Dict(d => Dict(freq_idx => zeros(Float64, 0, 6, 6, 0)) for d in depths),
            "dot_obs_gf_lag" => Dict(freq_idx => zeros(Float64, 0, 6, 0)),
        )
    end

    L = size(synamp_lag_lists[depths[1]][1], 3)
    nt_win = length(obs_win_list[1])

    # Pass 2: pre-allocate and fill
    obs_mat = zeros(Float64, n_entries, nt_win)
    obs_n2_vec = zeros(Float64, n_entries)
    dog_lag_arr = zeros(Float64, n_entries, 6, L)
    synamp_lag_arr = Dict{Float64, Array{Float64, 4}}()
    for d in depths
        synamp_lag_arr[d] = zeros(Float64, n_entries, 6, 6, L)
    end

    for i in 1:n_entries
        obs_mat[i, :] = obs_win_list[i]
        obs_n2_vec[i] = obs_n2_list[i]
        dog_lag_arr[i, :, :] = dog_lag_list[i]
        for d in depths
            synamp_lag_arr[d][i, :, :, :] = synamp_lag_lists[d][i]
        end
    end

    return Dict(
        "channel_id" => ch_vec,
        "station_idx" => sta_vec,
        "obs" => Dict(freq_idx => Dict("obs" => obs_mat, "obs_norm2" => obs_n2_vec)),
        "synamp_lag" => Dict(d => Dict(freq_idx => synamp_lag_arr[d]) for d in depths),
        "dot_obs_gf_lag" => Dict(freq_idx => dog_lag_arr),
    )
end
