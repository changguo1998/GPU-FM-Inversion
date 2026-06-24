#!/usr/bin/env julia
#
# tests/test_cross_lang.jl
#
# Cross-language verification suite.
# Part 1: MT consistency — Julia MTUtils vs C++ test_cross_lang
# Part 2: HDF5 round‑trip — Julia HDF5IO writes, C++ reads
# Part 3: Trial format — Julia generates trials, C++ reads
#
# Run:  julia tests/test_cross_lang.jl

include(joinpath(@__DIR__, "..", "shared", "mt", "src", "MT.jl"))
using .MT
using HDF5
using Random

const FORWARD_DIR = joinpath(@__DIR__, "..", "forward")
const TESTS_DIR = joinpath(FORWARD_DIR, "tests")
const CPP_BINARY = joinpath(TESTS_DIR, "test_cross_lang")
const TMP_DIR = mktempdir(; cleanup = true)

exit_failure = false

function run_cpp(mode, path; expected_code = 0)::Bool
    if !isfile(CPP_BINARY)
        println("SKIP: C++ binary not found: $CPP_BINARY")
        return false
    end
    cmd = Cmd(`$(CPP_BINARY) --mode $mode $path`)
    proc = run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = false)
    wait(proc)
    if proc.exitcode != expected_code
        println("FAIL: C++ $mode returned $(proc.exitcode), expected $expected_code")
        return false
    end
    return true
end

# ════════════════════════════════════════════════════════════════
# Part 1: MT CSV consistency
# ════════════════════════════════════════════════════════════════
function test_mt_csv()
    println("\n" * repeat("=", 60))
    println("Part 1: MT conversion consistency (Julia ↔ C++)")
    println(repeat("=", 60))

    n = 100
    rng = MersenneTwister(42)
    strikes = rand(rng, n) * 360.0
    dips = rand(rng, n) * 90.0
    rakes = (rand(rng, n) .- 0.5) * 180.0

    csv_path = joinpath(TMP_DIR, "mt_cross_lang.csv")
    println("Writing $n SDR combos → $csv_path")

    open(csv_path, "w") do io
        write(io, "strike,dip,rake,Mxx,Myy,Mzz,Mxy,Mxz,Myz\n")
        for i in 1:n
            mt = MT.sdr_to_mt(strikes[i], dips[i], rakes[i])
            write(
                io,
                join(
                    [strikes[i], dips[i], rakes[i], mt[1], mt[2], mt[3], mt[4], mt[5], mt[6]],
                    ',',
                ),
            )
            write(io, '\n')
        end
    end

    ok = run_cpp("mt-csv", csv_path)
    if ok
        println("Part 1: PASS")
    else
        global exit_failure = true
    end
    return ok
end

# ════════════════════════════════════════════════════════════════
# Part 2: HDF5 round‑trip
# ════════════════════════════════════════════════════════════════
function test_hdf5_roundtrip()
    println("\n" * repeat("=", 60))
    println("Part 2: HDF5 round‑trip (Julia writes → C++ reads)")
    println(repeat("=", 60))

    h5_path = joinpath(TMP_DIR, "cross_lang_hdf5.h5")
    println("Writing → $h5_path")

    h5open(h5_path, "w") do f
        gr = create_group(f, "cross_lang")

        # scalar int
        write(gr, "int_val", 42)

        # scalar double
        write(gr, "double_val", 3.14159)

        # 1D int
        write(gr, "int_array", Int32[10, 20, 30, 40, 50])

        # 1D double
        write(gr, "double_array", [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

        # 2D double (2×3).  HDF5.jl writes Julia column‑major ordering;
        # the C++ side accounts for this by transposing shape expectations.
        write(gr, "double_2d", [1.0 2.0 3.0; 4.0 5.0 6.0])
    end

    ok = run_cpp("hdf5", h5_path)
    if ok
        println("Part 2: PASS")
    else
        global exit_failure = true
    end
    return ok
end

# ════════════════════════════════════════════════════════════════
# Part 3: Trial format
# ════════════════════════════════════════════════════════════════
function test_trials()
    println("\n" * repeat("=", 60))
    println("Part 3: Trial generation format (Julia writes → C++ reads)")
    println(repeat("=", 60))

    n = 27  # 3×3×3 Cartesian product
    strikes = Float64[]
    dips = Float64[]
    rakes = Float64[]
    rng = MersenneTwister(7)

    for ist in 0:2, idp in 0:2, irk in 0:2
        push!(strikes, 10.0 + ist * 5.0)
        push!(dips, 30.0 + idp * 10.0)
        push!(rakes, -80.0 + irk * 70.0)
    end
    depth = fill(8.0, n)
    depth_idx = fill(Int32(2), n)
    freq_idx = repeat(Int32[0, 1], outer = div(n, 2) + 1)[1:n]

    h5_path = joinpath(TMP_DIR, "cross_lang_trials.h5")
    println("Writing $n trials → $h5_path")

    h5open(h5_path, "w") do f
        gr = create_group(f, "trials")
        write(gr, "strike", strikes)
        write(gr, "dip", dips)
        write(gr, "rake", rakes)
        write(gr, "depth", depth)
        write(gr, "depth_idx", depth_idx)
        write(gr, "freq_idx", freq_idx)
    end

    ok = run_cpp("trials", h5_path)
    if ok
        println("Part 3: PASS")
    else
        global exit_failure = true
    end
    return ok
end

# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

function main()
    println("TMP_DIR = $TMP_DIR")
    println("CPP_BINARY = $CPP_BINARY")
    println("CPP_BINARY exists? ", isfile(CPP_BINARY))

    results = [
        ("MT CSV consistency", test_mt_csv()),
        ("HDF5 round‑trip", test_hdf5_roundtrip()),
        ("Trial format verification", test_trials()),
    ]

    println("\n" * repeat("=", 60))
    println("RESULTS")
    println(repeat("=", 60))
    all_pass = true
    for (name, ok) in results
        status = ok ? "PASS" : "FAIL"
        println("  $status — $name")
        all_pass = all_pass && ok
    end

    if all_pass
        println("\nAll cross‑language tests passed.")
        exit(0)
    else
        println("\nSome tests FAILED. C++ binary: $CPP_BINARY")
        exit(1)
    end
end

main()
