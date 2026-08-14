"""Grid utilities: trial generation from strategy parameters and grid
refinement based on best-trial results (TrialGen + GridRefinement)."""
module Grid

# Can't `import IO` (clashes with Base.IO); load via PkgId + alias for sub-files.
const H5IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

include("trial_gen.jl")
include("grid_refinement.jl")

export generate_trials, default_grid
export TrialResult, refine_strategy, prompt_operator

end # module
