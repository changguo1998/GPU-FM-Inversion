# XCorr misfit plugin (template)
#
# Included inside Config.{name} (dynamically created inner module).
# Template for cross-correlation misfit — instantiated per phase via
# `Config.use_misfit!(:XcorrP, from = :Xcorr)`.
#
# Config stubs (user must override):
#   trim()     — time window [pre, post] seconds relative to arrival
#   maxlag_factor(), filter_order(), select_threshold(), deselect_threshold()

export trim, maxlag_factor, filter_order

export select_threshold,
    deselect_threshold, preprocess, process, is_freq_dependent, band_low, band_high

is_freq_dependent() = true
# -- Config namespace (user must override) --

function trim()::Vector{Float64}
    error("Xcorr.trim(): not implemented — return [-pre_sec, post_sec]  (e.g. [-2.0, 5.0])")
end

function maxlag_factor()::Float64
    error("Xcorr.maxlag_factor(): not implemented — return Float64  (e.g. 0.5)")
end

function filter_order()::Int
    error("Xcorr.filter_order(): not implemented — return Int  (e.g. 4)")
end

function band_low()::Vector{Int32}
    error("Xcorr.band_low(): not implemented — return Vector{Int32} of freq-band low indices")
end

function band_high()::Vector{Int32}
    error("Xcorr.band_high(): not implemented — return Vector{Int32} of freq-band high indices")
end

function select_threshold()::Float64
    error("Xcorr.select_threshold(): not implemented — return Float64  (e.g. 0.5)")
end

function deselect_threshold()::Float64
    error("Xcorr.deselect_threshold(): not implemented — return Float64  (e.g. 0.3)")
end

# -- Preprocessing --

const _Signal =
    Base.require(Base.PkgId(Base.UUID("c2443ae3-2a13-43e4-b75e-3c3d3ad453ec"), "Signal"))
const _IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

"""
    preprocess(obs, gf, dt, arrival_sample, low_cut, high_cut, window_factor;
               filter_order=4) -> (obs_proc, gf_proc, synamp, obs_norm2)

Bandpass filter + time-window trim for cross-correlation misfit.
"""
function preprocess(
    obs::Vector{Float64},
    gf::Matrix{Float64},
    dt::Float64,
    arrival_sample::Int,
    low_cut::Float64,
    high_cut::Float64,
    window_factor::Float64;
    filter_order::Int = 4,
)
    obs_filt = copy(obs)
    gf_filt = copy(gf)

    _Signal.bandpass_filter!(obs_filt, dt, low_cut, high_cut; order = filter_order)
    for c in 1:size(gf, 2)
        col = gf_filt[:, c]
        _Signal.bandpass_filter!(col, dt, low_cut, high_cut; order = filter_order)
        gf_filt[:, c] = col
    end

    obs_proc, gf_proc =
        _Signal.trim_time_window!(obs_filt, gf_filt, dt, arrival_sample, window_factor, high_cut)

    synamp = gf_proc' * gf_proc
    obs_norm2 = sum(obs_proc .^ 2)

    return obs_proc, gf_proc, synamp, obs_norm2
end

"""
    process(phases_pt, ptype, stations, picks, station_to_idx, channel_data,
            gf_data, depths, low_cut, high_cut, freq_idx, pf)

Batch preprocess XCorr for one phase type at one frequency band.
Returns a Dict mirroring the HDF5 schema:
  "channel_id"  => String[N_entries]
  "station_idx" => Int32[N_entries]
  "obs" => Dict(freq_idx => Dict("obs" => Float64[N_entries, N_samples],
                                 "obs_norm2" => Float64[N_entries]))
  "gf"  => Dict(depth => Dict(freq_idx => Dict("gf" => Float64[N_entries, 6, N_samples],
                                                "synamp" => Float64[N_entries, 6, 6])))
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
    low_cut::Float64,
    high_cut::Float64,
    freq_idx::Int,
    pf::Dict{String, Symbol},
)
    trim_win = trim()
    pre_sec = abs(trim_win[1])
    post_sec = abs(trim_win[2])
    wf_filter = max(pre_sec, post_sec) * high_cut
    filter_order_val = filter_order()

    # Pass 1: collect valid entries, determine nt_xc
    obs_list = Vector{Vector{Float64}}()
    obs_norm2_list = Float64[]
    gf_lists =
        Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    synamp_lists =
        Dict{Float64, Vector{Matrix{Float64}}}(d => Vector{Matrix{Float64}}() for d in depths)
    ch_vec = String[]
    sta_vec = Int32[]

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

        obs_proc, gf_proc0, synamp0, obs_n2 = preprocess(
            wf,
            gf_per_depth[depths[1]],
            dt,
            arrival_sample,
            low_cut,
            high_cut,
            wf_filter;
            filter_order = filter_order_val,
        )
        push!(obs_list, obs_proc)
        push!(obs_norm2_list, obs_n2)
        push!(gf_lists[depths[1]], gf_proc0)
        push!(synamp_lists[depths[1]], synamp0)
        push!(ch_vec, ch_id)
        push!(sta_vec, Int32(si))

        for depth_val in depths[2:end]
            _, gf_proc_d, synamp_d, _ = preprocess(
                wf,
                gf_per_depth[depth_val],
                dt,
                arrival_sample,
                low_cut,
                high_cut,
                wf_filter;
                filter_order = filter_order_val,
            )
            push!(gf_lists[depth_val], gf_proc_d)
            push!(synamp_lists[depth_val], synamp_d)
        end
    end

    n_entries = length(ch_vec)
    nt_xc = n_entries == 0 ? 0 : minimum(length.(obs_list))

    if n_entries == 0 || nt_xc == 0
        return Dict(
            "channel_id" => String[],
            "station_idx" => Int32[],
            "obs" =>
                Dict(freq_idx => Dict("obs" => zeros(Float64, 0, 0), "obs_norm2" => Float64[])),
            "gf" => Dict(
                d => Dict(
                    freq_idx => Dict(
                        "gf" => zeros(Float64, 0, 6, 0),
                        "synamp" => zeros(Float64, 0, 6, 6),
                    ),
                ) for d in depths
            ),
        )
    end

    # Pass 2: pre-allocate and fill
    obs_mat = zeros(Float64, n_entries, nt_xc)
    obs_n2_vec = zeros(Float64, n_entries)
    gf_arr = Dict{Float64, Array{Float64, 3}}()
    synamp_arr = Dict{Float64, Array{Float64, 3}}()
    for d in depths
        gf_arr[d] = zeros(Float64, n_entries, 6, nt_xc)
        synamp_arr[d] = zeros(Float64, n_entries, 6, 6)
    end

    for i in 1:n_entries
        obs_mat[i, :] = obs_list[i][1:nt_xc]
        obs_n2_vec[i] = obs_norm2_list[i]
        for d in depths
            gf_trimmed = gf_lists[d][i][1:nt_xc, :]
            gf_arr[d][i, :, :] = gf_trimmed'
            synamp_arr[d][i, :, :] = gf_trimmed' * gf_trimmed
        end
    end

    return Dict(
        "channel_id" => ch_vec,
        "station_idx" => sta_vec,
        "obs" => Dict(freq_idx => Dict("obs" => obs_mat, "obs_norm2" => obs_n2_vec)),
        "gf" => Dict(
            d => Dict(freq_idx => Dict("gf" => gf_arr[d], "synamp" => synamp_arr[d])) for
            d in depths
        ),
    )
end
