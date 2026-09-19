"""Compute squared natural-log S/P RMS residuals for aligned P/S entries."""
function psr_residual(
    obs_energy_p::AbstractVector,
    n_samples_p::AbstractVector{<:Integer},
    syn_energy_p::AbstractMatrix,
    obs_energy_s::AbstractVector,
    n_samples_s::AbstractVector{<:Integer},
    syn_energy_s::AbstractMatrix,
)
    n_entries, n_trials = size(syn_energy_p)
    size(syn_energy_s) == (n_entries, n_trials) ||
        throw(DimensionMismatch("P/S synthetic energy shapes differ"))
    length(obs_energy_p) == n_entries || throw(DimensionMismatch("P observation count differs"))
    length(obs_energy_s) == n_entries || throw(DimensionMismatch("S observation count differs"))
    length(n_samples_p) == n_entries || throw(DimensionMismatch("P sample count differs"))
    length(n_samples_s) == n_entries || throw(DimensionMismatch("S sample count differs"))

    result = fill(Inf, n_entries, n_trials)
    for entry in 1:n_entries
        ep_obs = obs_energy_p[entry]
        es_obs = obs_energy_s[entry]
        np = n_samples_p[entry]
        ns = n_samples_s[entry]
        (ep_obs > 0 && es_obs > 0 && np > 0 && ns > 0) || continue
        observed = log(sqrt((es_obs / ns) / (ep_obs / np)))
        for trial in 1:n_trials
            ep_syn = syn_energy_p[entry, trial]
            es_syn = syn_energy_s[entry, trial]
            (ep_syn > 0 && es_syn > 0) || continue
            synthetic = log(sqrt((es_syn / ns) / (ep_syn / np)))
            result[entry, trial] = abs2(observed - synthetic)
        end
    end
    return result
end

"""Compute per-entry L1 residuals of L2-normalized signed amplitudes."""
function normalized_polarity_residual(
    observed_signed::AbstractVector,
    synthetic_amplitude::AbstractMatrix,
    synthetic_sign::AbstractMatrix,
)
    size(synthetic_amplitude) == size(synthetic_sign) ||
        throw(DimensionMismatch("synthetic amplitude/sign shapes differ"))
    n_entries, n_trials = size(synthetic_amplitude)
    length(observed_signed) == n_entries ||
        throw(DimensionMismatch("observed/synthetic entry counts differ"))

    observed_norm = sqrt(sum(abs2, observed_signed))
    observed_norm > 0 || throw(ArgumentError("observed signed amplitudes have zero norm"))
    observed_normalized = observed_signed ./ observed_norm
    result = fill(Inf, n_entries, n_trials)
    for trial in 1:n_trials
        synthetic_energy = 0.0
        for entry in 1:n_entries
            signed = synthetic_amplitude[entry, trial] * synthetic_sign[entry, trial]
            synthetic_energy += abs2(signed)
        end
        synthetic_norm = sqrt(synthetic_energy)
        synthetic_norm > 0 || continue
        for entry in 1:n_entries
            signed = synthetic_amplitude[entry, trial] * synthetic_sign[entry, trial]
            result[entry, trial] = abs(observed_normalized[entry] - signed / synthetic_norm)
        end
    end
    return result
end
