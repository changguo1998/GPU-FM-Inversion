#!/usr/bin/env julia
#
# input.jl — Data ingestion and initialization stage
#
# Runs once before the main loop:
#   1. Read raw.h5 + config.toml
#   2. Preprocess all waveform data → database.h5
#   3. Write initial strategy → status_0.h5 (NO trials)
#
# Usage:
#   julia scripts/input.jl <raw.h5> <config.toml>

using HDF5
using TOML
using LinearAlgebra
using Dates
using Random

# ── Load shared modules ────────────────────────────────────────────────────────
SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "..", "shared", "io", "src", "IO.jl"))
using .IO
include(joinpath(SCRIPT_DIR, "..", "shared", "signal", "src", "Signal.jl"))
using .Signal

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parse_time_iso(t_str::String) -> Float64

Parse an ISO 8601 datetime string and return seconds since epoch.
Empty strings return NaN.
"""
function parse_time_iso(t_str::String)
    isempty(t_str) && return NaN
    return datetime2unix(DateTime(t_str))
end

"""
    haversine_distance(lat1, lon1, lat2, lon2) -> Float64

Compute great-circle distance (km) between two points on a sphere
(Earth radius = 6371 km).
"""
function haversine_distance(lat1, lon1, lat2, lon2)
    R = 6371.0
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    return 2 * R * asin(sqrt(a))
end

"""
    compute_azimuth(lat1, lon1, lat2, lon2) -> Float64

Compute azimuth (degrees, 0 = north, clockwise) from point 1 to point 2.
"""
function compute_azimuth(lat1, lon1, lat2, lon2)
    lat1r = deg2rad(lat1)
    lat2r = deg2rad(lat2)
    dlon = deg2rad(lon2 - lon1)
    x = sin(dlon) * cos(lat2r)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    az = rad2deg(atan(x, y))
    return mod(az, 360.0)
end

"""
    extract_station(phase_id::String) -> String

Extract station key from phase identifier.
"NET.ST1.Z.P" → "NET.ST1"
"""
function extract_station(phase_id::String)
    parts = split(phase_id, '.')
    return join(parts[1:2], '.')
end

"""
    extract_phase_type(phase_id::String) -> String

Extract phase type from phase identifier.
"NET.ST1.Z.P" → "P"
"""
function extract_phase_type(phase_id::String)
    parts = split(phase_id, '.')
    return parts[end]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main stage
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: julia scripts/input.jl <raw.h5> <config.toml>")
        exit(1)
    end

    raw_path = ARGS[1]
    config_path = ARGS[2]

    # Validate inputs
    if !isfile(raw_path)
        println(stderr, "ERROR: raw.h5 not found: $raw_path")
        exit(1)
    end
    if !isfile(config_path)
        println(stderr, "ERROR: config.toml not found: $config_path")
        exit(1)
    end

    println("[input] Reading raw.h5: $raw_path")
    println("[input] Reading config.toml: $config_path")

    # ── 1. Load config ─────────────────────────────────────────────────────────
    config = TOML.parsefile(config_path)
    @info "Config loaded" modules=config["misfit"]["modules"]

    # ── 2. Read raw.h5 ─────────────────────────────────────────────────────────
    event = IO.read_event(raw_path)
    picks = IO.read_phase_picks(raw_path)
    stations = IO.read_stations(raw_path)

    # Build station lookup: station_id → index in picks
    station_to_idx = Dict(pick.station_id => i for (i, pick) in enumerate(picks))

    n_phases = length(stations)
    n_stations_picks = length(picks)

    # Extract frequency bands
    freq_bands = config["freq_bands"]["bands"]  # e.g. [[0.5, 2.0]]
    n_frequencies = length(freq_bands)

    # Extract depth values
    depths = config["depths"]["values"]  # e.g. [5.0, 10.0, 15.0]
    n_depths = length(depths)

    # Extract misfit modules
    misfit_modules = config["misfit"]["modules"]  # ["XCorr", "Polarity", "PSR"]
    module_weights = Float64.(config["misfit"]["module_weights"])
    minimum_stations = config["misfit"]["minimum_stations"]

    # ── 3. Build /index ────────────────────────────────────────────────────────
    phase_ids = [s.id for s in stations]
    phase_types = [extract_phase_type(pid) for pid in phase_ids]

    # Map each phase to its station index (1-based, into picks array)
    station_idx = Int32[]
    for pid in phase_ids
        skey = extract_station(pid)
        push!(station_idx, Int32(get(station_to_idx, skey, 1)))
    end

    # Compute per-phase distance and azimuth from event to station
    distances = Float64[]
    azimuths = Float64[]
    for s in stations
        dist = haversine_distance(event.latitude, event.longitude, s.latitude, s.longitude)
        az = compute_azimuth(event.latitude, event.longitude, s.latitude, s.longitude)
        push!(distances, dist)
        push!(azimuths, az)
    end

    # greens_depth_idx: [N_phases × N_depths], one entry per phase per depth
    greens_depth_idx = Matrix{Int32}(undef, n_phases, n_depths)
    for p in 1:n_phases
        for d in 1:n_depths
            greens_depth_idx[p, d] = Int32(d)  # depth index = 1..n_depths
        end
    end

    index = IO.Index(
        phase_ids, phase_types, station_idx,
        distances, azimuths, greens_depth_idx
    )

    # ── 4. Preprocess waveforms ───────────────────────────────────────────────
    # Structure:
    # greens: Dict{phase_id => Dict{depth_idx => Matrix[Float64]}}  [N_samples × 6]
    # data: Dict{freq_idx => Dict{Symbol(module) => Dict{phase_id => ...}}}
    #
    # For each freq_idx and phase_id:
    #   XCorr: {obs, gf, synamp, obs_norm2}
    #   Polarity: {gf_pol, obs_pol}
    #   PSR: {amp_P, amp_S, obs_psr}  (station-level, shared across P and S phases)

    greens = Dict{String, Dict{Int32, Matrix{Float64}}}()
    data = Dict{Int, Dict{Symbol, Dict{String, Dict{String, Any}}}}()

    # Build P/S phase-pair lookup for PSR
    # Group phase_ids by station to find P/S pairs
    station_phases = Dict{String, Vector{String}}()
    for pid in phase_ids
        skey = extract_station(pid)
        ps = get!(Vector{String}, station_phases, skey)
        push!(ps, pid)
    end

    # Precompute P/S pairs for PSR
    psr_pairs = Tuple{String, String}[]  # (P_phase_id, S_phase_id)
    for (skey, phs) in station_phases
        p_phases = filter(p -> extract_phase_type(p) == "P", phs)
        s_phases = filter(p -> extract_phase_type(p) == "S", phs)
        for pp in p_phases, sp in s_phases
            push!(psr_pairs, (pp, sp))
        end
    end

    # Read xcorr config
    xcorr_cfg = get(config, "xcorr", Dict())
    maxlag_factor = get(xcorr_cfg, "maxlag_factor", 0.5)
    filter_order = get(xcorr_cfg, "filter_order", 4)
    P_trim = get(xcorr_cfg, "P_trim", [-2.0, 5.0])
    S_trim = get(xcorr_cfg, "S_trim", [-2.0, 5.0])

    # Read polarity config
    pol_cfg = get(config, "polarity", Dict())
    polarity_trim = get(pol_cfg, "trim", [0.0, 2.0])
    t_source = polarity_trim[2]  # 2 seconds for polarity window

    # Initialize data structure for each frequency
    for freq_idx in 1:n_frequencies
        bnd = freq_bands[freq_idx]
        low_cut = Float64(bnd[1])
        high_cut = Float64(bnd[2])

        data[freq_idx] = Dict{Symbol, Dict{String, Dict{String, Any}}}()
        data[freq_idx][:XCorr] = Dict{String, Dict{String, Any}}()
        data[freq_idx][:Polarity] = Dict{String, Dict{String, Any}}()
        data[freq_idx][:PSR] = Dict{String, Dict{String, Any}}()

        # ── Preprocess each phase ─────────────────────────────────────────────
        for (ph_idx, s) in enumerate(stations)
            pid = s.id
            ptype = extract_phase_type(pid)
            dt = s.dt

            # Read raw waveform
            wf = IO.read_waveform(raw_path, pid)
            n_samples = length(wf)

            # Determine arrival sample: get pick time for this station
            skey = extract_station(pid)
            st_idx = get(station_to_idx, skey, 1)
            pick = picks[st_idx]

            # Parse begin_time and pick time
            begin_unix = parse_time_iso(s.begin_time)
            if ptype == "P"
                pick_unix = parse_time_iso(pick.P_time)
                trim_cfg = P_trim
            else
                pick_unix = parse_time_iso(pick.S_time)
                trim_cfg = S_trim
            end

            if isnan(begin_unix) || isnan(pick_unix)
                arrival_sample = n_samples ÷ 2  # fallback: center
            else
                arrival_sample = round(Int, (pick_unix - begin_unix) / dt) + 1
                arrival_sample = clamp(arrival_sample, 1, n_samples)
            end

            # ── Generate or load Green's function for this phase × depth ─────
            if !haskey(greens, pid)
                greens[pid] = Dict{Int32, Matrix{Float64}}()
            end

            for (d_idx, depth_val) in enumerate(depths)
                didx = Int32(d_idx)
                if !haskey(greens[pid], didx)
                    # Generate synthetic GF for testing (or load from files in production)
                    gf_path = if haskey(config, "greens") && haskey(config["greens"], "gf_dir")
                        gf_dir = config["greens"]["gf_dir"]
                        joinpath(gf_dir, "$(pid)_depth$(d_idx).h5")
                    else
                        ""
                    end

                    if isfile(gf_path)
                        # Load from file
                        gf_mat = h5open(f -> read(f["greens"]), gf_path, "r")
                    else
                        # Generate synthetic GF: random matrix [N_samples × 6]
                        # with exponential decay to simulate realistic waveform
                        rng = Random.MersenneTwister(42 + d_idx + ph_idx)
                        gf_raw = randn(rng, n_samples, 6)
                        # Apply exponential decay envelope
                        decay = exp.(-(0:n_samples-1) ./ (n_samples / 4))
                        gf_mat = gf_raw .* decay
                    end
                    greens[pid][didx] = gf_mat
                end
            end

            # Use the first depth's GF for preprocessing (all depths use same obs)
            gf_full = greens[pid][Int32(1)]

            # Compute window factor from trim config
            pre_sec = abs(trim_cfg[1])
            post_sec = abs(trim_cfg[2])
            window_factor = max(pre_sec, post_sec) * high_cut

            # ── Module-specific preprocessing ─────────────────────────────────
            if "XCorr" in misfit_modules
                obs_proc, gf_proc, synamp, obs_norm2 = Signal.preprocess_xcorr!(
                    wf, gf_full, dt, arrival_sample,
                    low_cut, high_cut, window_factor;
                    filter_order=filter_order
                )
                data[freq_idx][:XCorr][pid] = Dict{String, Any}(
                    "obs" => obs_proc,
                    "gf" => gf_proc,
                    "synamp" => synamp,
                    "obs_norm2" => obs_norm2,
                )
            end

            if "Polarity" in misfit_modules && ptype == "P"
                pick_pol = picks[st_idx].P_polarity
                gf_pol, obs_pol = Signal.preprocess_polarity!(
                    gf_full, dt, arrival_sample, t_source, pick_pol
                )
                data[freq_idx][:Polarity][pid] = Dict{String, Any}(
                    "gf_pol" => gf_pol,
                    "obs_pol" => obs_pol,
                )
            end
        end

        # ── PSR preprocessing (pairs P and S for each station) ───────────────
        if "PSR" in misfit_modules
            for (p_pid, s_pid) in psr_pairs
                # Find the station info for P and S phases
                p_st = stations[findfirst(s -> s.id == p_pid, stations)]
                s_st = stations[findfirst(s -> s.id == s_pid, stations)]

                p_wf = IO.read_waveform(raw_path, p_pid)
                s_wf = IO.read_waveform(raw_path, s_pid)

                # Get P and S GF (first depth)
                gf_P = greens[p_pid][Int32(1)]
                gf_S = greens[s_pid][Int32(1)]

                dt = p_st.dt  # same dt for both
                skey = extract_station(p_pid)
                st_idx = get(station_to_idx, skey, 1)
                pick = picks[st_idx]

                begin_unix = parse_time_iso(p_st.begin_time)
                p_pick_unix = parse_time_iso(pick.P_time)
                s_pick_unix = parse_time_iso(pick.S_time)

                n_p_wf = length(p_wf)
                n_s_wf = length(s_wf)

                if isnan(begin_unix) || isnan(p_pick_unix)
                    arr_P = n_p_wf ÷ 2
                else
                    arr_P = round(Int, (p_pick_unix - begin_unix) / dt) + 1
                    arr_P = clamp(arr_P, 1, n_p_wf)
                end

                if isnan(begin_unix) || isnan(s_pick_unix)
                    arr_S = n_s_wf ÷ 2
                else
                    arr_S = round(Int, (s_pick_unix - begin_unix) / dt) + 1
                    arr_S = clamp(arr_S, 1, n_s_wf)
                end

                # PSR window parameters: use pre/post from trim
                pre_P = abs(P_trim[1])
                post_P = abs(P_trim[2])
                pre_S = abs(S_trim[1])
                post_S = abs(S_trim[2])

                amp_P, amp_S, obs_psr = Signal.preprocess_psr!(
                    p_wf, s_wf, gf_P, gf_S, dt,
                    arr_P, arr_S,
                    pre_P, post_P, pre_S, post_S
                )

                # Store PSR data keyed by {P_phase_id}_{S_phase_id}
                psr_key = "$(p_pid)|$(s_pid)"
                data[freq_idx][:PSR][psr_key] = Dict{String, Any}(
                    "amp_P" => amp_P,
                    "amp_S" => amp_S,
                    "obs_psr" => obs_psr,
                )
            end
        end
    end

    # ── 5. Build database config ───────────────────────────────────────────────
    db_config = Dict{String, Any}(
        "misfit_modules" => misfit_modules,
        "module_weights" => module_weights,
        "depth_vals" => Float64.(depths),
        "freq_bands_low" => Float64[low for (low, _) in freq_bands],
        "freq_bands_high" => Float64[high for (_, high) in freq_bands],
        "minimum_stations" => Int32(minimum_stations),
        "freq_test_max_iter" => Int32(get(config["freq_test"], "max_iter", 3)),
    )

    # Add per-module config sub-groups
    if "XCorr" in misfit_modules
        db_config["xcorr"] = Dict{String, Any}(
            "maxlag_factor" => Float64(maxlag_factor),
            "filter_order" => Int32(filter_order),
            "P_trim" => Float64.(P_trim),
            "S_trim" => Float64.(S_trim),
            "select_threshold" => Float64(get(xcorr_cfg, "select_threshold", 0.5)),
            "deselect_threshold" => Float64(get(xcorr_cfg, "deselect_threshold", 0.3)),
        )
    end

    if "Polarity" in misfit_modules
        db_config["polarity"] = Dict{String, Any}(
            "trim" => Float64.(polarity_trim),
        )
    end

    # ── 6. Write database.h5 ───────────────────────────────────────────────────
    db_path = "database.h5"
    println("[input] Writing database.h5")
    IO.write_database(db_path, greens, data, index, db_config)

    # ── 7. Write status_0.h5 with initial strategy ────────────────────────────
    grid = config["grid"]
    n_stations_for_mask = n_stations_picks  # number of stations from phase_picks

    strategy = IO.Strategy(
        Float64(grid["strike0"]),      # strike0
        Float64(grid["dstrike"]),      # dstrike
        Int32(grid["nstrike"]),        # nstrike
        Float64(grid["dip0"]),         # dip0
        Float64(grid["ddip"]),         # ddip
        Int32(grid["ndip"]),           # ndip
        Float64(grid["rake0"]),        # rake0
        Float64(grid["drake"]),        # drake
        Int32(grid["nrake"]),          # nrake
        Int32.(1:n_depths),            # depth_indices (all depths initially)
        Int32.(1:n_frequencies),       # freq_indices (all freq bands initially)
        ones(Int32, n_phases),         # xcorr_phase_mask (all active)
        ones(Int32, n_stations_for_mask),  # polarity_channel_mask (all active)
        ones(Int32, n_stations_for_mask),  # psr_channel_mask (all active)
        Float64.(module_weights),      # module_weights
        Float64[grid["strike0"], grid["dip0"], grid["rake0"]],  # best_sdr
        Int32(1),                      # best_depth_index
        Inf,                           # best_misfit
        Int32(0),                      # iteration
        Int32(0),                      # converged
        "",                            # convergence_reason
        zeros(Float64, n_frequencies, 3),  # freq_accumulated
        zeros(Float64, n_frequencies, get(config["freq_test"], "max_iter", 3)),  # freq_misfit_curve
        zeros(Float64, n_depths),      # depth_misfit_accumulated
    )

    status_path = "status_0.h5"
    println("[input] Writing status_0.h5")
    # Create fresh file for status_0
    h5open(status_path, "w") do f
        # Just create the file; write_strategy will use r+ mode
    end
    IO.write_strategy(status_path, strategy)

    # ── Summary ─────────────────────────────────────────────────────────────────
    println("\n[input] Stage complete:")
    println("  database.h5: /greens ($(length(greens)) phases × $n_depths depths)")
    println("  database.h5: /data ($n_frequencies frequencies × $(length(misfit_modules)) modules)")
    println("  database.h5: /config, /index")
    println("  status_0.h5: /strategy (initial grid, no trials)")
    println("  Phases: $(n_phases) | Stations: $(n_stations_picks) | Depths: $n_depths | Frequencies: $n_frequencies")
end

main()