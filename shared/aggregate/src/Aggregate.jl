module Aggregate

using Statistics

include("StdDev.jl")
include("extractors.jl")
include("composers.jl")
include("objective_primitives.jl")

export EXTRACTORS, COMPOSERS, psr_residual, normalized_polarity_residual

end # module Aggregate
