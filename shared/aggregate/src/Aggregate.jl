module Aggregate

using Statistics

include("StdDev.jl")
include("extractors.jl")
include("composers.jl")

export EXTRACTORS, COMPOSERS

end # module Aggregate
