# ─────────────────────────────────────────────────────────
# Grid Refinement
# ─────────────────────────────────────────────────────────

"""
    TrialResult

Aggregated best-trial information needed to compute the next iteration's
search grid.

# Fields
- `sdr::Vector{Float64}`: best [strike, dip, rake] in degrees
- `depth_idx::Int32`: index of best depth into `/config/depth_vals`
- `freq_idx::Int32`: index of best frequency band
- `misfit::Float64`: weighted misfit of best trial
- `depth_misfits::Vector{Float64}`: misfit per depth index at best SDR `[N_depths]`
- `freq_misfits::Vector{Float64}`: misfit per frequency index at best SDR `[N_frequencies]`
"""
struct TrialResult
    sdr::Vector{Float64}
    depth_idx::Int32
    freq_idx::Int32
    misfit::Float64
    depth_misfits::Vector{Float64}
    freq_misfits::Vector{Float64}
end

"""
    refine_strategy(current::H5IO.Strategy, best_trial::TrialResult) -> H5IO.Strategy

Compute the next iteration's search grid from the best trial result.

# Refinement Rules

- **Center**: new grid centered on best trial SDR
- **Step sizes**: halved — `new_step = old_step / 2`
- **Grid size**: always 3×3×3 SDR (`nstrike=3, ndip=3, nrake=3`)
- **Depth subset**: indices where `depth_misfit ≤ 1.2 × best_depth_misfit`
- **Frequency subset**: indices where `freq_misfit ≤ 1.2 × best_freq_misfit`
- **Empty subset**: fall back to single value (best index only)

Returns a new `H5IO.Strategy` with `converged=0` and `iteration` incremented.
The caller (assess.jl) prompts the operator and writes the strategy to
`status_{N+1}.h5`.
"""
function refine_strategy(current::H5IO.Strategy, best_trial::TrialResult)
    # ── SDR center ← best trial ────────────────────────
    new_strike0 = best_trial.sdr[1]
    new_dip0    = best_trial.sdr[2]
    new_rake0   = best_trial.sdr[3]

    # ── Halve step sizes ───────────────────────────────
    new_dstrike = current.dstrike / 2.0
    new_ddip    = current.ddip / 2.0
    new_drake   = current.drake / 2.0

    # ── Fixed 3×3×3 SDR grid ──────────────────────────
    new_nstrike = Int32(3)
    new_ndip    = Int32(3)
    new_nrake   = Int32(3)

    # ── Depth subset: within 20% of best depth misfit ──
    best_depth_misfit = best_trial.depth_misfits[best_trial.depth_idx]
    depth_threshold = best_depth_misfit * 1.2
    new_depth_indices = Int32[]
    for i in eachindex(best_trial.depth_misfits)
        if best_trial.depth_misfits[i] <= depth_threshold
            push!(new_depth_indices, Int32(i))
        end
    end
    if isempty(new_depth_indices)
        new_depth_indices = Int32[best_trial.depth_idx]
    end

    # ── Frequency subset: within 20% of best freq misfit ─
    best_freq_misfit = best_trial.freq_misfits[best_trial.freq_idx]
    freq_threshold = best_freq_misfit * 1.2
    new_freq_indices = Int32[]
    for i in eachindex(best_trial.freq_misfits)
        if best_trial.freq_misfits[i] <= freq_threshold
            push!(new_freq_indices, Int32(i))
        end
    end
    if isempty(new_freq_indices)
        new_freq_indices = Int32[best_trial.freq_idx]
    end

    # ── Accumulate depth misfits (element-wise min) ────
    if isempty(current.depth_misfit_accumulated)
        new_depth_misfit_accumulated = copy(best_trial.depth_misfits)
    else
        new_depth_misfit_accumulated = min.(current.depth_misfit_accumulated,
                                            best_trial.depth_misfits)
    end

    # ── Build output Strategy ──────────────────────────
    return H5IO.Strategy(
        new_strike0, new_dstrike, new_nstrike,
        new_dip0, new_ddip, new_ndip,
        new_rake0, new_drake, new_nrake,
        new_depth_indices,
        new_freq_indices,
        copy(current.xcorr_phase_mask),
        copy(current.polarity_channel_mask),
        copy(current.psr_channel_mask),
        copy(current.module_weights),
        Float64[best_trial.sdr[1], best_trial.sdr[2], best_trial.sdr[3]],
        best_trial.depth_idx,
        best_trial.misfit,
        current.iteration + Int32(1),
        Int32(0),    # converged = 0 (not yet converged; set by assess.jl on operator break)
        "",          # convergence_reason (set by assess.jl)
        copy(current.freq_accumulated),
        copy(current.freq_misfit_curve),
        new_depth_misfit_accumulated,
    )
end

# ─────────────────────────────────────────────────────────
# Operator Prompt
# ─────────────────────────────────────────────────────────

"""
    prompt_operator(best_sdr, misfit, current::H5IO.Strategy; io_in=stdin, io_out=stdout) -> Bool

Display current best result and grid, then ask the operator whether to
continue to the next iteration.

Returns `true` for "y" / "Y", `false` for anything else.
"""
function prompt_operator(best_sdr, misfit, current::H5IO.Strategy;
                         io_in::Base.IO=stdin, io_out::Base.IO=stdout)
    println(io_out)
    println(io_out, "Best SDR: (strike=$(best_sdr[1]), dip=$(best_sdr[2]), rake=$(best_sdr[3])), Misfit=$misfit")

    # ── Format current grid description ────────────────
    parts = String[]
    if current.nstrike > 0
        push!(parts, "strike=$(current.strike0)±$(current.dstrike)°")
    else
        push!(parts, "strike=$(current.strike0)°")
    end
    if current.ndip > 0
        push!(parts, "dip=$(current.dip0)±$(current.ddip)°")
    else
        push!(parts, "dip=$(current.dip0)°")
    end
    if current.nrake > 0
        push!(parts, "rake=$(current.rake0)±$(current.drake)°")
    else
        push!(parts, "rake=$(current.rake0)°")
    end
    println(io_out, "Current grid: $(join(parts, ", "))")

    print(io_out, "Continue? [y/N] ")
    flush(io_out)

    answer = strip(readline(io_in))
    return lowercase(answer) == "y"
end