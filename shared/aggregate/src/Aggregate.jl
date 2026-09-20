module Aggregate

using Statistics

include("StdDev.jl")
include("extractors.jl")
include("composers.jl")
include("objective_primitives.jl")
include("objective_aggregation.jl")

export EXTRACTORS, COMPOSERS, aggregate_objectives, normalize_objective
export objective_trial_means, psr_residual, normalized_polarity_residual

end # module Aggregate
