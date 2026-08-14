# Grid Refinement

"""
    TrialResult — best-trial info to build the next iteration's search grid.

# Fields
- `sdr::Vector{Float64}`: best [strike, dip, rake] deg
- `depth_idx::Int32` / `freq_idx::Int32`: best indices into `/paraspace`
- `misfit::Float64`: weighted misfit of best trial
- `depth_misfits` / `freq_misfits::Vector{Float64}`: per-index misfit at best SDR
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

Next-iteration search grid centered on best trial: SDR steps halved, fixed
3×3×3 grid; depth/freq subsets keep indices with misfit ≤ 1.2 × best (single
best index if empty). Returns a `Strategy` with `converged=0`, `iteration+1`
(the caller assess.jl prompts the operator and writes `status_{N+1}.h5`).
"""
function refine_strategy(current::H5IO.Strategy, best_trial::TrialResult)::H5IO.Strategy
    new_dstrike = current.dstrike / 2.0
    new_ddip = current.ddip / 2.0
    new_drake = current.drake / 2.0

    # 3×3×3 grid centered on best: expand_axis(var0,d,3) = [var0,var0+d,var0+2d],
    # so var0 = best - d, clamped to strike [0,360), dip [0,90], rake [-90,90].
    new_strike0 = mod(best_trial.sdr[1] - new_dstrike, 360.0)
    new_dip0 = clamp(best_trial.sdr[2] - new_ddip, 0.0, 90.0)
    new_rake0 = clamp(best_trial.sdr[3] - new_drake, -90.0, 90.0)

    new_nstrike = Int32(3)
    new_ndip = Int32(3)
    new_nrake = Int32(3)

    # Depth subset: within 20% of best depth misfit
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

    # Frequency subset: within 20% of best freq misfit
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


    return H5IO.Strategy(
        new_strike0,
        new_dstrike,
        new_nstrike,
        new_dip0,
        new_ddip,
        new_ndip,
        new_rake0,
        new_drake,
        new_nrake,
        new_depth_indices,
        new_freq_indices,
        current.iteration + Int32(1),
    )
end

# Operator Prompt

"""prompt_operator(best_sdr, misfit, current; io_in=stdin, io_out=stdout) -> Bool

Print the best result and current grid, ask whether to continue;
returns true for "y"/"Y", false otherwise."""
function prompt_operator(
    best_sdr,
    misfit,
    current::H5IO.Strategy;
    io_in::Base.IO = stdin,
    io_out::Base.IO = stdout,
)
    :Bool
    println(io_out)
    println(
        io_out,
        "Best SDR: (strike=$(best_sdr[1]), dip=$(best_sdr[2]), rake=$(best_sdr[3])), Misfit=$misfit",
    )

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
