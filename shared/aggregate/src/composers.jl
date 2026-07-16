# Composers: aggregate Level 1 base misfit matrices into Level 2 misfit.
# Keyed by aggregate operator name. Returns Dict{Symbol, Matrix} keyed by
# the operator's outputs(), so assess.jl selects the requested output field.
const COMPOSERS = Dict{Symbol, Function}()

# StdDev: per-station standard deviation + mean across base misfits.
# base_misfits[i]: Matrix{Float64} (N_phases_i × N_trials)
# base_station_idx[i]: Vector{Int32} mapping each phase row -> 1-based station
# Returns Dict(:relative_offset => std, :mean => mean), each [N_stations × N_trials]
COMPOSERS[:StdDev] =
    function (base_misfits::Vector{Matrix{Float64}}, base_station_idx::Vector{Vector{Int32}}, ctx)
        N_stations = ctx.N_stations
        N_trials = size(base_misfits[1], 2)
        std_out = fill(NaN, N_stations, N_trials)
        mean_out = fill(NaN, N_stations, N_trials)
        for s in 1:N_stations, t in 1:N_trials
            vals = Float64[]
            for (i, m) in enumerate(base_misfits)
                for (p, si) in enumerate(base_station_idx[i])
                    si == s && push!(vals, m[p, t])
                end
            end
            if length(vals) >= 2
                std_out[s, t] = std(vals)
                mean_out[s, t] = mean(vals)
            end
        end
        return Dict(:relative_offset => std_out, :mean => mean_out)
    end
