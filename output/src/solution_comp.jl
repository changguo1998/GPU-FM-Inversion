module SolutionComp

using Statistics: std

"""
    compute_depth_range(
        depth_vals::AbstractVector{Float64},
        depth_misfit_vec::AbstractVector{Float64},
        tolerance::Float64=0.05,
    ) -> Vector{Float64}

Compute depth range by applying tolerance to per-depth misfit accumulated by assess.jl.

# Returns
- `[min_depth, max_depth]` — depths within `tolerance` fraction of the best misfit.
"""
function compute_depth_range(
    depth_vals::AbstractVector{Float64},
    depth_misfit_vec::AbstractVector{Float64};
    tolerance::Float64 = 0.05,
)
    if length(depth_misfit_vec) == 0
        return [NaN, NaN]
    end

    valid = filter(!isnan, depth_misfit_vec)
    if isempty(valid)
        return [NaN, NaN]
    end
    best_misfit = minimum(valid)
    threshold = best_misfit * (1.0 + tolerance)

    valid_depths = Float64[]
    for i in eachindex(depth_misfit_vec)
        if !isnan(depth_misfit_vec[i]) && depth_misfit_vec[i] <= threshold
            push!(valid_depths, depth_vals[i])
        end
    end

    if isempty(valid_depths)
        return [NaN, NaN]
    end

    return [minimum(valid_depths), maximum(valid_depths)]
end

"""
    compute_sdr_std(freq_accumulated::AbstractMatrix{Float64}) -> (Float64, Float64, Float64)

Compute standard deviation of strike, dip, rake across frequency bands.

`freq_accumulated` has shape `[N_frequencies, 3]` where each row is `[strike, dip, rake]`.

# Returns
- `(strike_std, dip_std, rake_std)`
"""
function compute_sdr_std(freq_accumulated::AbstractMatrix{Float64})
    if size(freq_accumulated, 1) <= 1
        return (NaN, NaN, NaN)
    end
    s_std = std(freq_accumulated[:, 1])
    d_std = std(freq_accumulated[:, 2])
    r_std = std(freq_accumulated[:, 3])
    return (s_std, d_std, r_std)
end

"""
    compile_solution(
        strategy::Strategy,
        trials::TrialSet,
        misfits::Dict{Symbol, Matrix{Float64}},
        index,
        config::Dict{String, Any},
        status_dir::String;
        synthesize_waveforms::Bool=false,
    ) -> NamedTuple

Compile final focal mechanism solution from converged pipeline results.

# Arguments
- `strategy`: final strategy from status file (with converged=1)
- `trials`: trial table from the final status file
- `misfits`: raw misfit matrices from the final status file
- `index`: phase index from database.h5
- `config`: config dict from database.h5
- `status_dir`: directory containing status files (for status discovery)
- `synthesize_waveforms`: if true, compute synthetic seismograms

# Returns
- `(solution, uncertainty, per_phase, per_station_summary, summary)` NamedTuple of Dicts
"""
function compile_solution(
    strategy,
    trials,
    misfits,
    index,
    config;
    synthesize_waveforms::Bool = false,
)
    n_trials = length(trials.strike)
    n_phases = length(index.phase_ids)
    n_stations = length(unique(index.station_idx))

    # ── 1. Re-aggregate misfits to find best trial ──
    module_weights = strategy.module_weights
    if length(module_weights) < 3
        module_weights = [module_weights; zeros(Float64, 3 - length(module_weights))]
    end

    xcorr_data = get(misfits, :xcorr, zeros(Float64, n_phases, n_trials))
    polarity_data = get(misfits, :polarity, zeros(Float64, n_stations, n_trials))
    psr_data = get(misfits, :psr, zeros(Float64, n_stations, n_trials))

    xcorr_mask = strategy.xcorr_phase_mask
    if length(xcorr_mask) < n_phases
        xcorr_mask = ones(Int32, n_phases)
    end
    pol_mask = strategy.polarity_channel_mask
    if length(pol_mask) < n_stations
        pol_mask = ones(Int32, n_stations)
    end
    psr_mask = strategy.psr_channel_mask
    if length(psr_mask) < n_stations
        psr_mask = ones(Int32, n_stations)
    end

    # Use AssessUtils.aggregate_misfits if available, otherwise fallback
    if @isdefined(AssessUtils) && isdefined(Main, :AssessUtils)
        total, best_idx, per_module_scores = Main.AssessUtils.aggregate_misfits(
            xcorr_data, polarity_data, psr_data,
            xcorr_mask, pol_mask, psr_mask,
            module_weights,
        )
    else
        total, best_idx, per_module_scores = _fallback_aggregate(
            xcorr_data, polarity_data, psr_data,
            xcorr_mask, pol_mask, psr_mask,
            module_weights,
        )
    end

    best_strike = trials.strike[best_idx]
    best_dip = trials.dip[best_idx]
    best_rake = trials.rake[best_idx]
    best_depth = trials.depth[best_idx]
    best_misfit = total[best_idx]

    # Verify best SDR matches strategy (within tolerance)
    strategy_best = strategy.best_sdr
    @assert isapprox(best_strike, strategy_best[1]; atol=1.0) ||
            abs(best_strike - strategy_best[1]) < 1e-3 "Best strike mismatch: $(best_strike) vs $(strategy_best[1])"
    @assert isapprox(best_dip, strategy_best[2]; atol=1.0) ||
            abs(best_dip - strategy_best[2]) < 1e-3 "Best dip mismatch: $(best_dip) vs $(strategy_best[2])"
    @assert isapprox(best_rake, strategy_best[3]; atol=1.0) ||
            abs(best_rake - strategy_best[3]) < 1e-3 "Best rake mismatch: $(best_rake) vs $(strategy_best[3])"

    # ── 2. SDR → MT conversion ──
    mt = Main.MTUtils.sdr_to_mt(best_strike, best_dip, best_rake)

    solution = Dict{String, Any}(
        "strike" => best_strike,
        "dip" => best_dip,
        "rake" => best_rake,
        "depth" => best_depth,
        "moment_tensor" => mt,
        "misfit" => best_misfit,
    )

    # ── 3. Frequency uncertainty ──
    freq_accumulated = getfield(strategy, :freq_accumulated)
    strike_std_val, dip_std_val, rake_std_val = compute_sdr_std(freq_accumulated)
    freq_misfit_curve = getfield(strategy, :freq_misfit_curve)

    # ── 4. Depth range ──
    depth_vals_from_config = get(config, "depth_vals", Float64[])
    if isempty(depth_vals_from_config) && haskey(config, "depth")
        # Some configs nest depth_vals under a "depth" or top-level key
        dcfg = config["depth"]
        if dcfg isa Vector
            depth_vals_from_config = dcfg
        elseif dcfg isa Dict && haskey(dcfg, "vals")
            depth_vals_from_config = dcfg["vals"]
        end
    end
    depth_misfit_vec = getfield(strategy, :depth_misfit_accumulated)
    depth_range = compute_depth_range(depth_vals_from_config, depth_misfit_vec)

    uncertainty = Dict{String, Any}(
        "strike_std" => strike_std_val,
        "dip_std" => dip_std_val,
        "rake_std" => rake_std_val,
        "depth_range" => depth_range,
        "freq_test_misfit_curve" => freq_misfit_curve,
    )

    # ── 5. Per-phase breakdown ──
    phase_ids = index.phase_ids
    phase_types = index.phase_type
    station_indices = index.station_idx
    n_stations = length(unique(station_indices))

    # Build station and channel identifiers from phase_ids
    station_ids = [split(pid, ".")[1] * "." * split(pid, ".")[2] for pid in phase_ids]
    channel_ids = [join(split(pid, ".")[1:3], ".") for pid in phase_ids]

    # Extract per-phase misfits at best trial
    misfit_per_module = zeros(Float64, 3, n_phases)
    xc = get(misfits, :xcorr, zeros(Float64, n_phases, n_trials))
    pol = get(misfits, :polarity, zeros(Float64, n_stations, n_trials))
    psr = get(misfits, :psr, zeros(Float64, n_stations, n_trials))

    for ph in 1:n_phases
        misfit_per_module[1, ph] = xc[ph, best_idx]
    end
    # Polarity/PSR are per-station; map station_idx back to phase entries
    for ph in 1:n_phases
        si = station_indices[ph]
        if 1 <= si <= size(pol, 1)
            misfit_per_module[2, ph] = pol[si, best_idx]
        end
        if 1 <= si <= size(psr, 1)
            misfit_per_module[3, ph] = psr[si, best_idx]
        end
    end

    # Selected phases: those where XCorr phase mask is active
    selected = xcorr_mask
    if length(selected) < n_phases
        selected = ones(Int32, n_phases)
    end

    # Cross-correlation values: use XCorr misfit at best trial (1.0 - misfit → CC)
    cross_correlation = [1.0 - misfit_per_module[1, ph] for ph in 1:n_phases]

    per_phase = Dict{String, Any}(
        "phase_id" => phase_ids,
        "channel_id" => channel_ids,
        "station_id" => station_ids,
        "phase_type" => phase_types,
        "misfit_per_module" => misfit_per_module,
        "selected" => selected,
        "cross_correlation" => cross_correlation,
    )

    # ── 6. Per-station summary ───────────────────────────────────────────────────
    # Group by station_id to compute station-level aggregates
    unique_stations = unique(station_ids)
    n_unique = length(unique_stations)

    sta_summary_station_ids = String[]
    sta_summary_n_channels = Int32[]
    sta_summary_n_phases = Int32[]
    sta_summary_mean_cc = Float64[]
    sta_summary_pol_match = Int32[]
    sta_summary_misfit_total = Float64[]

    for sta in unique_stations
        # Find phases belonging to this station
        idx_in_sta = findall(s -> s == sta, station_ids)
        n_ph_in_sta = length(idx_in_sta)

        # Unique channels for this station
        channels_in_sta = unique([channel_ids[i] for i in idx_in_sta])
        n_ch = length(channels_in_sta)

        # Mean cross-correlation across this station's phases
        cc_vals = [cross_correlation[i] for i in idx_in_sta]
        mean_cc = sum(cc_vals) / length(cc_vals)

        # Polarity match count: count phases where polarity misfit == 0
        pol_match = 0
        for i in idx_in_sta
            si = station_indices[i]
            if 1 <= si <= size(pol, 1) && pol[si, best_idx] == 0.0
                pol_match += 1
            end
        end

        # Misfit total: sum of all module misfits at best trial for this station's phases
        total_misfit = 0.0
        for i in idx_in_sta
            for m in 1:3
                v = misfit_per_module[m, i]
                if !isnan(v)
                    total_misfit += v
                end
            end
        end

        push!(sta_summary_station_ids, sta)
        push!(sta_summary_n_channels, Int32(n_ch))
        push!(sta_summary_n_phases, Int32(n_ph_in_sta))
        push!(sta_summary_mean_cc, mean_cc)
        push!(sta_summary_pol_match, Int32(pol_match))
        push!(sta_summary_misfit_total, total_misfit)
    end

    per_station_summary = Dict{String, Any}(
        "station_id" => sta_summary_station_ids,
        "n_channels" => sta_summary_n_channels,
        "n_phases" => sta_summary_n_phases,
        "mean_cross_correlation" => sta_summary_mean_cc,
        "polarity_match" => sta_summary_pol_match,
        "misfit_total" => sta_summary_misfit_total,
    )

    # ── 7. Summary ──
    convergence_reason = getfield(strategy, :convergence_reason)
    total_iterations = getfield(strategy, :iteration)

    summary = Dict{String, Any}(
        "total_iterations" => total_iterations,
        "total_trials" => n_trials,
        "convergence_reason" => convergence_reason,
    )

    return (solution = solution, uncertainty = uncertainty,
            per_phase = per_phase, per_station_summary = per_station_summary, summary = summary)
end

# ─── Fallback aggregator (when AssessUtils not loaded) ───
function _fallback_aggregate(
    xcorr::AbstractMatrix{Float64},
    polarity::AbstractMatrix{Float64},
    psr::AbstractMatrix{Float64},
    xcorr_phase_mask::AbstractVector{<:Integer},
    polarity_channel_mask::AbstractVector{<:Integer},
    psr_channel_mask::AbstractVector{<:Integer},
    weights::AbstractVector{Float64},
)
    n_trials = size(xcorr, 2)

    function masked_sum(data, mask)
        scores = zeros(Float64, n_trials)
        n_rows = size(data, 1)
        for i in 1:n_rows
            if i <= length(mask) && mask[i] == 0
                continue
            end
            for j in 1:n_trials
                v = data[i, j]
                if !isnan(v)
                    scores[j] += v
                end
            end
        end
        return scores
    end

    xc = masked_sum(xcorr, xcorr_phase_mask) .* weights[1]
    pl = masked_sum(polarity, polarity_channel_mask) .* weights[2]
    pr = masked_sum(psr, psr_channel_mask) .* weights[3]
    total = xc .+ pl .+ pr

    if all(isnan, total)
        error("all trials NaN")
    end

    best_idx = 0
    best_val = Inf
    for j in 1:n_trials
        if !isnan(total[j]) && total[j] < best_val
            best_val = total[j]
            best_idx = j
        end
    end

    per_module = Dict(:xcorr => xc, :polarity => pl, :psr => pr)
    return total, best_idx, per_module
end

export compile_solution, compute_depth_range, compute_sdr_std

end # module SolutionComp