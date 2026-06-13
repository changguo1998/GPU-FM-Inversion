#!/usr/bin/env julia
#
# test_input_stage.jl — Integration test for input stage
#
# 1. Generate synthetic raw.h5 + config.toml
# 2. Run input stage
# 3. Verify database.h5 and status_0.h5 contents
#
# Usage: julia --project=input input/test/test_input_stage.jl

using Test
using HDF5
using TOML

# ── Include synthetic data generator ────────────────────────────────────────
include(joinpath(@__DIR__, "..", "..", "tests", "synthetic_data.jl"))

# ── Include HDF5IO for verification ─────────────────────────────────────────
include(joinpath(@__DIR__, "..", "..", "shared", "HDF5IO.jl", "src", "HDF5IO.jl"))
using .HDF5IO

# ── Setup: generate synthetic data ──────────────────────────────────────────
test_dir = mktempdir(; cleanup=true)
@info "Test directory: $test_dir"

raw_h5 = joinpath(test_dir, "raw.h5")
config_toml = joinpath(test_dir, "config.toml")

# synthetic_data.jl already uses its own write paths; we need to run it in test_dir
# and then reference those files. But synthetic_data.jl writes to CWD or given dir.
# We need a fresh approach: generate manually.
include_string = """
using HDF5, Random, Dates
mkpath("$test_dir")
Random.seed!(42)

# /event
h5open("$raw_h5", "w") do f
    g = create_group(f, "/event")
    g["longitude"] = 120.0; g["latitude"] = 30.0
    g["depth"] = 10.0; g["magnitude"] = 5.0
    g["origintime"] = "2024-01-01T00:00:00"

    g = create_group(f, "/phase_picks")
    g["station_ids"] = ["NET.ST1", "NET.ST2", "NET.ST3"]
    g["P_time"] = ["2024-01-01T00:00:10", "2024-01-01T00:00:12", "2024-01-01T00:00:14"]
    g["S_time"] = ["2024-01-01T00:00:18", "2024-01-01T00:00:21", "2024-01-01T00:00:24"]
    g["P_polarity"] = Int8[1, -1, 0]

    g = create_group(f, "/stations")
    g["id"] = ["NET.ST1.Z.P","NET.ST1.Z.S","NET.ST2.Z.P","NET.ST2.Z.S","NET.ST3.Z.P","NET.ST3.Z.S"]
    g["network"] = fill("NET", 6)
    g["station"] = ["ST1","ST1","ST2","ST2","ST3","ST3"]
    g["component"] = fill("Z", 6)
    g["latitude"] = [30.5, 30.5, 29.5, 29.5, 30.0, 30.0]
    g["longitude"] = [120.5, 120.5, 119.5, 119.5, 120.0, 120.0]
    g["elevation"] = [500.0, 500.0, 600.0, 600.0, 550.0, 550.0]
    g["dt"] = fill(0.01, 6)
    g["begin_time"] = fill("2024-01-01T00:00:05", 6)

    g = create_group(f, "/waveforms")
    for id in ["NET.ST1.Z.P","NET.ST1.Z.S","NET.ST2.Z.P","NET.ST2.Z.S","NET.ST3.Z.P","NET.ST3.Z.S"]
        g[id] = randn(Float64, 2000)
    end
end

# config.toml
write("$config_toml", \"\"\"
[misfit]
modules = ["XCorr", "Polarity", "PSR"]
module_weights = [0.5, 0.25, 0.25]
minimum_stations = 2

[freq_bands]
bands = [[0.5, 2.0]]
n_frequencies = 1

[grid]
strike0 = 45.0
dstrike = 20.0
nstrike = 3
dip0 = 30.0
ddip = 20.0
ndip = 3
rake0 = 0.0
drake = 20.0
nrake = 3

[depths]
values = [5.0, 10.0, 15.0]
n_depths = 3

[greens]
gf_dir = ""
model = "synthetic"

[xcorr]
maxlag_factor = 0.5
filter_order = 4
P_trim = [-2.0, 5.0]
S_trim = [-2.0, 5.0]
select_threshold = 0.5
deselect_threshold = 0.3

[polarity]
trim = [0.0, 2.0]

[freq_test]
max_iter = 3
\"\"\")
"""

run(`$(Base.julia_cmd()) -e "$include_string"`)
@test isfile(raw_h5)
@test isfile(config_toml)

# ── Run input stage ─────────────────────────────────────────────────────────
cd(test_dir) do
    input_script = joinpath(@__DIR__, "..", "src", "input.jl")
    cmd = `$(Base.julia_cmd()) --project=$(@__DIR__)/.. $input_script $raw_h5 $config_toml`
    run(cmd)
end

# ── Verify database.h5 ──────────────────────────────────────────────────────
db_path = joinpath(test_dir, "database.h5")
@test isfile(db_path)

@test h5exists(db_path, "greens")
@test h5exists(db_path, "data")
@test h5exists(db_path, "config")
@test h5exists(db_path, "index")

# Check /index contents
idx = read_index(db_path)
@test length(idx.phase_ids) == 6
@test all(pid -> occursin(".Z.", pid), idx.phase_ids)
@test idx.phase_type == ["P", "S", "P", "S", "P", "S"]

# Check /greens: 6 phases × 3 depths = 18 matrices
h5open(db_path, "r") do f
    grp = f["greens"]
    phase_count = length(keys(grp))
    @test phase_count == 6
    for pid in keys(grp)
        pg = grp[pid]
        @test length(keys(pg)) == 3  # 3 depths
        for d in keys(pg)
            gf = read(pg[d])
            @test size(gf) == (2000, 6)  # N_samples × 6
        end
    end
end

# Check /data: 1 freq × modules × phases
h5open(db_path, "r") do f
    data_grp = f["data"]
    @test length(keys(data_grp)) == 1  # 1 frequency band
    for fidx in keys(data_grp)
        fg = data_grp[fidx]
        module_names = keys(fg)
        @test "XCorr" in module_names
        @test "Polarity" in module_names
        @test "PSR" in module_names

        # XCorr: 6 phases
        xg = fg["XCorr"]
        @test length(keys(xg)) >= 6
        for pid in keys(xg)
            phd = xg[pid]
            @test "obs" in keys(phd)
            @test "gf" in keys(phd)
            @test "synamp" in keys(phd)
            @test "obs_norm2" in keys(phd)
            @test size(read(phd["synamp"])) == (6, 6)
        end

        # Polarity: P phases only (3)
        pg = fg["Polarity"]
        p_phases = filter(p -> endswith(p, ".P"), keys(pg))
        @test length(p_phases) == 3
        for pid in p_phases
            phd = pg[pid]
            @test "gf_pol" in keys(phd)
            @test "obs_pol" in keys(phd)
        end
    end
end

# Check /config
cfg = read_config(db_path)
@test "misfit_modules" in keys(cfg)
@test "depth_vals" in keys(cfg)
@test "freq_bands_low" in keys(cfg)
@test "freq_bands_high" in keys(cfg)

# ── Verify status_0.h5 ──────────────────────────────────────────────────────
status_path = joinpath(test_dir, "status_0.h5")
@test isfile(status_path)
@test h5exists(status_path, "strategy")

# Check strategy contents
strat = read_strategy(status_path)
@test strat.strike0 ≈ 45.0
@test strat.dip0 ≈ 30.0
@test strat.rake0 ≈ 0.0
@test strat.nstrike == 3
@test strat.ndip == 3
@test strat.nrake == 3
@test strat.iteration == 0
@test strat.converged == 0
@test strat.best_misfit == Inf
@test length(strat.xcorr_phase_mask) == 6
@test length(strat.depth_indices) == 3
@test length(strat.freq_indices) == 1

# Check status_0.h5 does NOT have /trials
@test !h5exists(status_path, "trials")

@info "All tests passed!"