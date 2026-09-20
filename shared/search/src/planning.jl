# Budget-constrained search planning

"""Selected global parameter indices for one search batch."""
struct SearchPlan
    strike_indices::Vector{Int32}
    dip_indices::Vector{Int32}
    rake_indices::Vector{Int32}
    depth_indices::Vector{Int32}
    freq_indices::Vector{Int32}
    duration_indices::Vector{Int32}
    levels::NTuple{6, Int32}
end

function _initial_positions(n::Int, periodic::Bool)::Vector{Int32}
    n <= 1 && return Int32[1]
    if periodic
        return Int32[1, div(n, 2) + 1]
    end
    return Int32[1, n]
end

function _refine_positions(indices::Vector{Int32}, n::Int, periodic::Bool)::Vector{Int32}
    n <= 1 && return Int32[1]
    refined = copy(indices)
    for i in 1:(length(indices) - 1)
        gap = Int(indices[i + 1] - indices[i])
        gap > 1 && push!(refined, indices[i] + div(gap, 2))
    end
    if periodic
        gap = Int(indices[1]) + n - Int(indices[end])
        if gap > 1
            midpoint = mod1(Int(indices[end]) + div(gap, 2), n)
            push!(refined, Int32(midpoint))
        end
    end
    sort!(unique!(refined))
    return refined
end

function _maximum_gap(indices::Vector{Int32}, n::Int, periodic::Bool)::Int
    n <= 1 && return 0
    gap = maximum(diff(Int.(indices)); init = 0)
    periodic && (gap = max(gap, Int(indices[1]) + n - Int(indices[end])))
    return gap
end

function _resolution(indices::Vector{Int32}, n::Int, periodic::Bool)::Float64
    denominator = periodic ? n : max(n - 1, 1)
    return _maximum_gap(indices, n, periodic) / denominator
end

function _trial_count(indices::NTuple{6, Vector{Int32}})::Int
    return prod(length.(indices))
end

function _axis_values(values::Vector{Int32}, positions::Vector{Int32})::Vector{Int32}
    isempty(values) && return Int32[1]
    return values[positions]
end

"""Create a deterministic search plan whose Cartesian product does not exceed `budget`.

The planner starts from boundary samples on ordered axes and antipodal samples
on periodic strike. It greedily bisects the axis with the largest normalized
resolution improvement while the resulting trial count fits the budget.
"""
function budgeted_plan(strategy::H5IO.Strategy, budget::Integer)::SearchPlan
    budget > 0 || throw(ArgumentError("trial budget must be positive"))
    lengths = (
        max(Int(strategy.nstrike), 1),
        max(Int(strategy.ndip), 1),
        max(Int(strategy.nrake), 1),
        max(length(strategy.depth_indices), 1),
        max(length(strategy.freq_indices), 1),
        max(length(strategy.duration_indices), 1),
    )
    periodic = (true, false, false, false, false, false)
    positions = ntuple(i -> _initial_positions(lengths[i], periodic[i]), 6)
    levels = ntuple(_ -> Int32(0), 6)
    minimum_trials = _trial_count(positions)
    budget < minimum_trials && throw(
        ArgumentError("trial budget $budget is below minimum coarse sample count $minimum_trials"),
    )

    while true
        best_axis = 0
        best_positions = positions
        best_gain = 0.0
        best_count = typemax(Int)
        for axis in 1:6
            candidate_axis = _refine_positions(positions[axis], lengths[axis], periodic[axis])
            length(candidate_axis) == length(positions[axis]) && continue
            candidate = ntuple(i -> i == axis ? candidate_axis : positions[i], 6)
            candidate_count = _trial_count(candidate)
            candidate_count > budget && continue
            gain =
                _resolution(positions[axis], lengths[axis], periodic[axis]) -
                _resolution(candidate_axis, lengths[axis], periodic[axis])
            if gain > best_gain || (gain == best_gain && candidate_count < best_count)
                best_axis = axis
                best_positions = candidate
                best_gain = gain
                best_count = candidate_count
            end
        end
        best_axis == 0 && break
        positions = best_positions
        levels = ntuple(i -> i == best_axis ? levels[i] + Int32(1) : levels[i], 6)
    end

    return SearchPlan(
        positions[1],
        positions[2],
        positions[3],
        _axis_values(strategy.depth_indices, positions[4]),
        _axis_values(strategy.freq_indices, positions[5]),
        _axis_values(strategy.duration_indices, positions[6]),
        levels,
    )
end

"""Generate the Cartesian product selected by a `SearchPlan`."""
function generate_trials(plan::SearchPlan)::H5IO.TrialSet
    return _generate_trials(
        plan.strike_indices,
        plan.dip_indices,
        plan.rake_indices,
        plan.depth_indices,
        plan.freq_indices,
        plan.duration_indices,
    )
end
