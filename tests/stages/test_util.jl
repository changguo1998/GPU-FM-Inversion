# test_util.jl — shared helpers for stage (pipeline) tests.
#
# Stage scripts are flat top-level scripts (no `main()` wrapper, global Config
# state), so tests run each stage as an isolated Julia subprocess against a
# freshly generated synthetic data directory, then assert on the output HDF5
# files. These helpers are the common plumbing for that flow.

const PROJECT_ROOT = dirname(dirname(@__DIR__))  # tests/stages -> project root

"""
    run_cmd(cmd) -> (ok, code, stdout, stderr)

Run a shell command, capturing combined output. Never throws on failure.
"""
function run_cmd(cmd::Cmd)
    out = IOBuffer()
    err = IOBuffer()
    p = run(pipeline(ignorestatus(cmd), stdout = out, stderr = err))
    return (
        ok = success(p),
        code = p.exitcode,
        stdout = String(take!(out)),
        stderr = String(take!(err)),
    )
end

"""
    run_stage_script(script, args; env) -> (ok, code, output)

Run a Julia stage script as a subprocess in the project environment.
Optional `env` pairs (e.g. "DATA_DIR" => dir) are added to the child env.
"""
function run_stage_script(script::AbstractString, args::Vector{String}; env = nothing)
    cmd = `$(Base.julia_cmd()) --project=$PROJECT_ROOT $(joinpath(PROJECT_ROOT, script)) $(args)`
    env === nothing || (cmd = addenv(cmd, env))
    return run_cmd(cmd)
end

"""
    make_synthetic(dir; nsta, stf_sigma, seed_strike, seed_dip, seed_rake)

Generate a fresh synthetic event (stations.txt, phases.txt, *.dat) into `dir`
using `tests/synthetic_data.jl`. Deterministic (fixed RNG seeds).
"""
function make_synthetic(
    dir::AbstractString;
    nsta::Int = 3,
    stf_sigma::Float64 = 0.2,
    strike::Float64 = 30.0,
    dip::Float64 = 60.0,
    rake::Float64 = 90.0,
)
    mkpath(dir)
    gen = joinpath(PROJECT_ROOT, "tests", "synthetic_data.jl")
    cmd = `$(Base.julia_cmd()) $gen $dir --nsta $nsta --stf-sigma $stf_sigma --strike $strike --dip $dip --rake $rake`
    r = run_cmd(cmd)
    @assert r.ok "synthetic_data.jl failed: $(r.stderr)"
    return r
end

# Physical SDR -> MT conversion, matching tests/synthetic_data.jl (NED).
function sdr_to_mt(s::Float64, d::Float64, r::Float64)::Vector{Float64}
    sd = sind(d);
    cd = cosd(d);
    ss = sind(s);
    cs = cosd(s);
    sr = sind(r);
    cr = cosd(r)
    Mxx = -(sd * cr * sind(2s) + sin(2d) * sr * ss^2)
    Myy = sd * cr * sind(2s) - sin(2d) * sr * cs^2
    Mzz = sin(2d) * sr
    Mxy = sd * cr * cosd(2s) + 0.5 * sin(2d) * sr * sind(2s)
    Mxz = -(cd * cr * cs + cosd(2d) * sr * ss)
    Myz = -(cd * cr * ss - cosd(2d) * sr * cs)
    return [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
end
