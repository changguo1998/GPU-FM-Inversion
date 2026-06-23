"""
Grid utilities: trial generation from strategy parameters and
grid refinement based on best-trial results.

Combines TrialGen (generate trials from search grid) and
GridRefinement (compute next iteration's grid from best trial).
"""
module Grid

# Include IO types needed by grid_refinement
include(joinpath(@__DIR__, "..", "..", "io", "src", "IO.jl"))
using .IO

include("trial_gen.jl")
include("grid_refinement.jl")

export generate_trials, TrialSet, GridStrategy
export TrialResult, refine_strategy, prompt_operator

end # module