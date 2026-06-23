#!/usr/bin/env julia
#
# tests/test_mt_consistency.jl
#
# Cross-language verification: Julia MTUtils.sdr_to_mt vs C++ test_mt_to_csv.
#
# Run:  julia tests/test_mt_consistency.jl

include(joinpath(@__DIR__, "..", "shared", "mt", "src", "MT.jl"))
using .MT
using Random

const TEST_BINARY = joinpath(@__DIR__, "..", "forward", "tests", "test_mt_to_csv")

if !isfile(TEST_BINARY)
    error("C++ test binary not found at $TEST_BINARY")
end

function run_test()
    rng = MersenneTwister(42)
    n = 100
    strikes = rand(rng, n) * 360.0           # [0, 360)
    dips    = rand(rng, n) * 90.0            # [0, 90]
    rakes   = (rand(rng, n) .- 0.5) * 180.0   # [-90, 90]

    println("Running cross-language consistency test: Julia vs C++ ($n trials)")
    println("Binary: $TEST_BINARY")
    println()

    max_diff = 0.0
    all_pass = true

    for i in 1:n
        s, d, r = strikes[i], dips[i], rakes[i]
        julia_mt = MT.sdr_to_mt(s, d, r)

        cmd = `"$(TEST_BINARY)" $(s) $(d) $(r)`
        lines = readlines(pipeline(cmd))
        @assert length(lines) == 2 "Expected 2 lines, got: $lines"

        data = split(lines[2], ',')
        cpp_mt = [parse(Float64, data[j]) for j in 4:9]

        for j in 1:6
            diff = abs(julia_mt[j] - cpp_mt[j])
            if diff > max_diff
                max_diff = diff
            end
            if diff >= 1e-6
                all_pass = false
                @warn "Mismatch at trial $i, component $j" sdr=(s,d,r) julia=julia_mt[j] cpp=cpp_mt[j] diff=diff
            end
        end
    end

    println()
    if all_pass && max_diff < 1e-6
        println("PASS: max diff = $(max_diff) (< 1e-6)")
    else
        println("FAIL: max diff = $(max_diff) (>= 1e-6)")
        exit(1)
    end
end

run_test()