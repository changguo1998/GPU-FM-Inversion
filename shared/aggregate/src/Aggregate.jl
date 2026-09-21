module Aggregate

using Statistics

include("StdDev.jl")
include("extractors.jl")
include("composers.jl")
include("objective_primitives.jl")
include("hierarchical_sum.jl")

export EXTRACTORS, COMPOSERS, hierarchical_sum
export psr_residual, normalized_polarity_residual

end # module Aggregate
