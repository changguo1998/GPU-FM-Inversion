module Misfit

export AbstractExpr, P, S, ampScale, decode_expression, encode_expression, energy
export evaluate, input, lagCC, maxCC, observed, rms, signScale, synthetic

include("Expressions.jl")
include("Operators.jl")

module Xcorr
include("Xcorr.jl")
end

module Polarity
include("Polarity.jl")
end

module Psr
include("Psr.jl")
end

end # module Misfit
