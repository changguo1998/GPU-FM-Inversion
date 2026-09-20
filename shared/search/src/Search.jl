"""Search-space utilities for defining parameter axes and generating trials."""
module Search

# Can't `import IO` (clashes with Base.IO); load via PkgId + alias for sub-files.
const H5IO = Base.require(Base.PkgId(Base.UUID("4a4c5d4c-b010-4bf7-8ff7-4f9ab209ee1d"), "IO"))

include("trial_gen.jl")
include("planning.jl")

export generate_trials, default_grid
export SearchPlan, budgeted_plan

end # module
