"""Average finite entries for every trial."""
function objective_trial_means(misfit::AbstractMatrix{<:Real})
    result = fill(NaN, size(misfit, 2))
    for trial in axes(misfit, 2)
        values = filter(isfinite, misfit[:, trial])
        isempty(values) || (result[trial] = sum(values) / length(values))
    end
    return result
end

"""Map trial-level objective values to [0, 1] using min-max scaling."""
function normalize_objective(values::AbstractVector{<:Real})
    finite_values = filter(isfinite, values)
    isempty(finite_values) && throw(ArgumentError("objective has no finite trial values"))
    low = minimum(finite_values)
    high = maximum(finite_values)
    high == low && return zeros(Float64, length(values))
    result = Vector{Float64}(undef, length(values))
    for i in eachindex(values)
        result[i] = isfinite(values[i]) ? clamp((values[i] - low) / (high - low), 0.0, 1.0) : 1.0
    end
    return result
end

"""Min-max normalize each objective, then average objectives per trial."""
function aggregate_objectives(misfits::AbstractDict; absolute_modules = Set())
    normalized = Dict{eltype(keys(misfits)), Vector{Float64}}()
    total = nothing
    count = 0
    for (name, matrix) in misfits
        values = objective_trial_means(matrix)
        name in absolute_modules && (values = abs.(values))
        scaled = normalize_objective(values)
        normalized[name] = scaled
        total = total === nothing ? scaled : total .+ scaled
        count += 1
    end
    count > 0 || throw(ArgumentError("no objectives to aggregate"))
    return normalized, total ./ count
end
