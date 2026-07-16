# Output extractors: transform raw intermediates (Dict of arrays) into
# Level 1 misfit matrices. Keyed by (operator, output_field).
const EXTRACTORS = Dict{Tuple{Symbol, Symbol}, Function}()

# XCorr: cc_max -> 1 - cc_max (normalized CC misfit)
EXTRACTORS[(:Xcorr, :cc_max)] = (inter, ctx) -> 1.0 .- inter["cc_max"]

# XCorr: best_lag -> best_lag * dt (absolute time shift in seconds)
EXTRACTORS[(:Xcorr, :best_lag)] = (inter, ctx) -> Float64.(inter["best_lag"]) .* ctx.dt

# Polarity: syn_sign -> 0/1 mismatch vs observed polarity
EXTRACTORS[(:Polarity, :syn_sign)] =
    (inter, ctx) -> Float64.(Int8.(inter["syn_sign"]) .!= ctx.obs_pol)

# Polarity: dot_value -> |dot| (confidence weight)
EXTRACTORS[(:Polarity, :dot_value)] = (inter, ctx) -> abs.(inter["dot_value"])
