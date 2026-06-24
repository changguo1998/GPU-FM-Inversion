"""
Grid utilities: trial generation from strategy parameters and
grid refinement based on best-trial results.

Combines TrialGen (generate trials from search grid) and
GridRefinement (compute next iteration's grid from best trial).
"""
module Grid

# Cannot `import IO` — name clashes with Base.IO.
# Load via PkgId to disambiguate, then alias for use in sub-files.
const H5IO = Base.require(Base.PkgId(
    Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

include("trial_gen.jl")
include("grid_refinement.jl")

export generate_trials, TrialSet, GridStrategy
export TrialResult, refine_strategy, prompt_operator

end # module