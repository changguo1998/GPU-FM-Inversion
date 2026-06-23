#!/usr/bin/env julia
#
# input.jl — Data ingestion and initialization stage
#
# Runs once before the main loop:
#   1. Read raw.h5 + config.jl
#   2. Preprocess all waveform data → database.h5
#   3. Write initial strategy → status_0.h5 (NO trials)
#
# Usage:
#   julia scripts/input.jl <raw.h5> <config.jl>

using HDF5
using LinearAlgebra
using Dates
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# Logging: writes to both stdout and input.log
# ═══════════════════════════════════════════════════════════════════════════════

const LOG_FILE = open("input.log", "w")

function log_info(msg::String)
    line = "[input] $(msg)"
    println(line)
    println(LOG_FILE, line)
    flush(LOG_FILE)
end

function log_warn(msg::String)
    line = "[input] WARN: $(msg)"
    println(stderr, line)
    println(LOG_FILE, line)
    flush(LOG_FILE)
end

function log_error(msg::String)
    line = "[input] ERROR: $(msg)"
    println(stderr, line)
    println(LOG_FILE, line)
    flush(LOG_FILE)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Load shared modules
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = @__DIR__

include(joinpath(SCRIPT_DIR, "..", "shared", "io", "src", "IO.jl"))
using .IO

include(joinpath(SCRIPT_DIR, "..", "shared", "signal", "src", "Signal.jl"))
using .Signal

include(joinpath(SCRIPT_DIR, "..", "shared", "config", "src", "Config.jl"))
using .Config

# ═══════════════════════════════════════════════════════════════════════════════
# CLI (no file-existence checks — driver handles validation)
# ═══════════════════════════════════════════════════════════════════════════════

raw_path   = ARGS[1]
config_jl  = ARGS[2]

# ── Log header ──
log_info("─"^70)
log_info("input stage started")
log_info("  raw.h5   = $raw_path")
log_info("  config   = $config_jl")
log_info("  log file = input.log")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load config (user script defines Config.* interface functions)
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Loading config: $config_jl")
include(abspath(config_jl))  # defines Config.misfit_modules(), Config.grid_params(), …

# ── Read config values via interface ──
misfit_modules    = Config.misfit_modules()
module_weights    = Config.module_weights()
minimum_stations  = Config.minimum_stations()
freq_bands        = Config.freq_bands()
depths            = Config.depths()
grid              = Config.grid_params()
xcorr             = Config.xcorr_params()
polarity          = Config.polarity_params()
greens_cfg        = Config.greens_params()
freq_test_maxiter = Config.freq_test_max_iter()

n_frequencies = length(freq_bands)
n_depths      = length(depths)

log_info("Config loaded")
log_info("  misfit_modules   = $misfit_modules")
log_info("  module_weights   = $module_weights")
log_info("  freq_bands       = $freq_bands")
log_info("  depths           = $depths")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Read raw.h5
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Reading raw.h5 …")
event    = IO.read_event(raw_path)
picks    = IO.read_phase_picks(raw_path)
stations = IO.read_stations(raw_path)

station_to_idx = Dict(pick.station_id => i for (i, pick) in enumerate(picks))

n_phases          = length(stations)
n_stations_picks  = length(picks)

log_info("  event      = (lon=$(event.longitude), lat=$(event.latitude), depth=$(event.depth), M=$(event.magnitude))")
log_info("  phases     = $n_phases")
log_info("  stations   = $n_stations_picks")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build /index
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Building index …")

phase_ids   = [s.id for s in stations]
phase_types = [IO.extract_phase_type(pid) for pid in phase_ids]

station_idx = Int32[]
for pid in phase_ids
    skey = IO.extract_station(pid)
    push!(station_idx, Int32(get(station_to_idx, skey, 1)))
end

distances = Float64[]
azimuths  = Float64[]
for s in stations
    push!(distances, IO.haversine_distance(event.latitude, event.longitude, s.latitude, s.longitude))
    push!(azimuths,  IO.compute_azimuth(event.latitude, event.longitude, s.latitude, s.longitude))
end

greens_depth_idx = Matrix{Int32}(undef, n_phases, n_depths)
for p in 1:n_phases, d in 1:n_depths
    greens_depth_idx[p, d] = Int32(d)
end

index = IO.Index(
    phase_ids, phase_types, station_idx,
    distances, azimuths, greens_depth_idx
)

log_info("  index built ($(length(phase_ids)) phases, $n_depths depths)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Preprocess waveforms
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Preprocessing waveforms …")

greens = Dict{String, Dict{Int32, Matrix{Float64}}}()
data   = Dict{Int, Dict{Symbol, Dict{String, Dict{String, Any}}}}()

# Build P/S phase-pair lookup for PSR
station_phases = Dict{String, Vector{String}}()
for pid in phase_ids
    skey = IO.extract_station(pid)
    push!(get!(Vector{String}, station_phases, skey), pid)
end

psr_pairs = Tuple{String, String}[]
for (skey, phs) in station_phases
    p_phases = filter(p -> IO.extract_phase_type(p) == "P", phs)
    s_phases = filter(p -> IO.extract_phase_type(p) == "S", phs)
    for pp in p_phases, sp in s_phases
        push!(psr_pairs, (pp, sp))
    end
end

maxlag_factor       = xcorr.maxlag_factor
filter_order        = xcorr.filter_order
P_trim              = xcorr.P_trim
S_trim              = xcorr.S_trim
select_threshold    = xcorr.select_threshold
deselect_threshold  = xcorr.deselect_threshold
polarity_trim       = polarity.trim
t_source            = polarity_trim[2]
gf_dir              = greens_cfg.gf_dir

for freq_idx in 1:n_frequencies
    bnd      = freq_bands[freq_idx]
    low_cut  = Float64(bnd[1])
    high_cut = Float64(bnd[2])

    log_info("  freq band $freq_idx/$n_frequencies: [$low_cut, $high_cut] Hz")

    data[freq_idx] = Dict{Symbol, Dict{String, Dict{String, Any}}}()
    data[freq_idx][:XCorr]    = Dict{String, Dict{String, Any}}()
    data[freq_idx][:Polarity] = Dict{String, Dict{String, Any}}()
    data[freq_idx][:PSR]      = Dict{String, Dict{String, Any}}()

    for (ph_idx, s) in enumerate(stations)
        pid   = s.id
        ptype = IO.extract_phase_type(pid)
        dt    = s.dt

        wf       = IO.read_waveform(raw_path, pid)
        n_samples = length(wf)

        skey  = IO.extract_station(pid)
        st_idx = get(station_to_idx, skey, 1)
        pick  = picks[st_idx]

        begin_unix = IO.parse_time_iso(s.begin_time)
        if ptype == "P"
            pick_unix = IO.parse_time_iso(pick.P_time)
            trim_cfg  = P_trim
        else
            pick_unix = IO.parse_time_iso(pick.S_time)
            trim_cfg  = S_trim
        end

        if isnan(begin_unix) || isnan(pick_unix)
            arrival_sample = n_samples ÷ 2
        else
            arrival_sample = clamp(round(Int, (pick_unix - begin_unix) / dt) + 1, 1, n_samples)
        end

        if !haskey(greens, pid)
            greens[pid] = Dict{Int32, Matrix{Float64}}()
        end

        for (d_idx, depth_val) in enumerate(depths)
            didx = Int32(d_idx)
            if !haskey(greens[pid], didx)
                gf_path = isempty(gf_dir) ? "" : joinpath(gf_dir, "$(pid)_depth$(d_idx).h5")

                if isfile(gf_path)
                    gf_mat = h5open(f -> read(f["greens"]), gf_path, "r")
                else
                    rng    = Random.MersenneTwister(42 + d_idx + ph_idx)
                    gf_raw = randn(rng, n_samples, 6)
                    decay  = exp.(-(0:n_samples-1) ./ (n_samples / 4))
                    gf_mat = gf_raw .* decay
                end
                greens[pid][didx] = gf_mat
            end
        end

        gf_full      = greens[pid][Int32(1)]
        pre_sec      = abs(trim_cfg[1])
        post_sec     = abs(trim_cfg[2])
        window_factor = max(pre_sec, post_sec) * high_cut

        if "XCorr" in misfit_modules
            obs_proc, gf_proc, synamp, obs_norm2 = Signal.preprocess_xcorr!(
                wf, gf_full, dt, arrival_sample,
                low_cut, high_cut, window_factor;
                filter_order=filter_order
            )
            data[freq_idx][:XCorr][pid] = Dict{String, Any}(
                "obs" => obs_proc, "gf" => gf_proc,
                "synamp" => synamp, "obs_norm2" => obs_norm2,
            )
        end

        if "Polarity" in misfit_modules && ptype == "P"
            pick_pol = picks[st_idx].P_polarity
            gf_pol, obs_pol = Signal.preprocess_polarity!(
                gf_full, dt, arrival_sample, t_source, pick_pol
            )
            data[freq_idx][:Polarity][pid] = Dict{String, Any}(
                "gf_pol" => gf_pol, "obs_pol" => obs_pol,
            )
        end
    end

    # ── PSR preprocessing ──
    if "PSR" in misfit_modules
        for (p_pid, s_pid) in psr_pairs
            p_st = stations[findfirst(s -> s.id == p_pid, stations)]
            s_st = stations[findfirst(s -> s.id == s_pid, stations)]

            p_wf = IO.read_waveform(raw_path, p_pid)
            s_wf = IO.read_waveform(raw_path, s_pid)
            gf_P = greens[p_pid][Int32(1)]
            gf_S = greens[s_pid][Int32(1)]

            dt    = p_st.dt
            skey  = IO.extract_station(p_pid)
            st_idx = get(station_to_idx, skey, 1)
            pick  = picks[st_idx]

            begin_unix  = IO.parse_time_iso(p_st.begin_time)
            p_pick_unix = IO.parse_time_iso(pick.P_time)
            s_pick_unix = IO.parse_time_iso(pick.S_time)

            n_p_wf = length(p_wf)
            n_s_wf = length(s_wf)

            arr_P = if isnan(begin_unix) || isnan(p_pick_unix)
                n_p_wf ÷ 2
            else
                clamp(round(Int, (p_pick_unix - begin_unix) / dt) + 1, 1, n_p_wf)
            end
            arr_S = if isnan(begin_unix) || isnan(s_pick_unix)
                n_s_wf ÷ 2
            else
                clamp(round(Int, (s_pick_unix - begin_unix) / dt) + 1, 1, n_s_wf)
            end

            pre_P  = abs(P_trim[1]); post_P = abs(P_trim[2])
            pre_S  = abs(S_trim[1]); post_S = abs(S_trim[2])

            amp_P, amp_S, obs_psr = Signal.preprocess_psr!(
                p_wf, s_wf, gf_P, gf_S, dt,
                arr_P, arr_S, pre_P, post_P, pre_S, post_S
            )

            psr_key = "$(p_pid)|$(s_pid)"
            data[freq_idx][:PSR][psr_key] = Dict{String, Any}(
                "amp_P" => amp_P, "amp_S" => amp_S, "obs_psr" => obs_psr,
            )
        end
    end
end

log_info("  preprocessing complete ($n_frequencies freqs, $(length(misfit_modules)) modules)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Build database config
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Building database config …")

db_config = Dict{String, Any}(
    "misfit_modules"     => misfit_modules,
    "module_weights"     => module_weights,
    "depth_vals"         => Float64.(depths),
    "freq_bands_low"     => Float64[low  for (low, _) in freq_bands],
    "freq_bands_high"    => Float64[high for (_, high) in freq_bands],
    "minimum_stations"   => Int32(minimum_stations),
    "freq_test_max_iter" => Int32(freq_test_maxiter),
)

if "XCorr" in misfit_modules
    db_config["xcorr"] = Dict{String, Any}(
        "maxlag_factor"      => Float64(maxlag_factor),
        "filter_order"       => Int32(filter_order),
        "P_trim"             => Float64.(P_trim),
        "S_trim"             => Float64.(S_trim),
        "select_threshold"   => Float64(select_threshold),
        "deselect_threshold" => Float64(deselect_threshold),
    )
end

if "Polarity" in misfit_modules
    db_config["polarity"] = Dict{String, Any}(
        "trim" => Float64.(polarity_trim),
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Write database.h5
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Writing database.h5 …")
IO.write_database("database.h5", greens, data, index, db_config)
log_info("  database.h5 written")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Write status_0.h5
# ═══════════════════════════════════════════════════════════════════════════════

log_info("Writing status_0.h5 …")

strategy = IO.Strategy(
    Float64(grid.strike0),
    Float64(grid.dstrike),
    Int32(grid.nstrike),
    Float64(grid.dip0),
    Float64(grid.ddip),
    Int32(grid.ndip),
    Float64(grid.rake0),
    Float64(grid.drake),
    Int32(grid.nrake),
    Int32.(1:n_depths),
    Int32.(1:n_frequencies),
    ones(Int32, n_phases),
    ones(Int32, n_stations_picks),
    ones(Int32, n_stations_picks),
    Float64.(module_weights),
    Float64[grid.strike0, grid.dip0, grid.rake0],
    Int32(1),
    Inf,
    Int32(0),
    Int32(0),
    "",
    zeros(Float64, n_frequencies, 3),
    zeros(Float64, n_frequencies, freq_test_maxiter),
    zeros(Float64, n_depths),
)

h5open("status_0.h5", "w") do f end  # create fresh file
IO.write_strategy("status_0.h5", strategy)
log_info("  status_0.h5 written")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

log_info("")
log_info("Stage complete:")
log_info("  database.h5  : /greens ($(length(greens)) phases × $n_depths depths)")
log_info("  database.h5  : /data ($n_frequencies frequencies × $(length(misfit_modules)) modules)")
log_info("  database.h5  : /config, /index")
log_info("  status_0.h5  : /strategy (initial grid, no trials)")
log_info("  Phases: $n_phases | Stations: $n_stations_picks | Depths: $n_depths | Freqs: $n_frequencies")
log_info("")
log_info("─"^70)

close(LOG_FILE)