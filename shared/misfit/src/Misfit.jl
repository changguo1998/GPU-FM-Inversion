module Misfit

export AbstractExpr, P, S, ampScale, decode_expression, encode_expression, energy
export evaluate, evaluate_pipeline, input, lagCC, maxCC, observed, rms, signScale, synthetic

include("Expressions.jl")
include("Operators.jl")
include("PipelineEvaluator.jl")

module Xcorr
include("Xcorr.jl")
end

module Polarity
include("Polarity.jl")
end

module Psr
include("Psr.jl")
end

module Expression
include("Expression.jl")
end

end # module Misfit
