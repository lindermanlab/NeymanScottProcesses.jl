struct AnnealedSampler <: AbstractSampler
    verbose::Bool
    temperatures::Vector{Real}
    anneal_method::Symbol
    subsampler::AbstractSampler
end

valid_save_keys(S::AnnealedSampler) = valid_save_keys(S.subsampler)

function AnnealedSampler(
    subsampler::AbstractSampler, max_temp::Real, anneal_method::Symbol; 
    verbose=true, num_samples=10
)
    temps = exp10.(range(log10(max_temp), 0, length=num_samples))
    return AnnealedSampler(verbose, temps, anneal_method, subsampler)
end

function Base.getproperty(obj::AnnealedSampler, sym::Symbol)
    if sym === :save_interval
        return 1
    elseif sym === :num_samples
        return length(obj.temperatures)
    elseif sym === :save_keys
        return obj.subsampler.save_keys
    else
        return getfield(obj, sym)
    end
end

"""
Run annealed sampling.
"""
function (S::AnnealedSampler)(
    model::NeymanScottModel, 
    data::Vector;
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    # Sampler options.
    verbose = S.verbose
    temperatures = S.temperatures
    subsampler = S.subsampler
    anneal_method = S.anneal_method

    # Set up annealing function
    true_priors = deepcopy(model.priors)
    
    # Initialize assignments and results
    assignments = initialize_assignments(data, initial_assignments)
    results = initialize_results(model, assignments, S)

    for temp in temperatures
        verbose && println("TEMP:  ", temp)

        # Anneal and recompute restaurant process probabilities
        model.priors = anneal(true_priors, temp, anneal_method)

        # Resample global variables, conditioned on new priors.
        # Importantly, this updates `globals.bkgd_log_prob` and
        # `globals.bkgd_rate` which are sensitive to annealing.
        gibbs_sample_globals!(
            model.globals, model.domain, model.priors, data, assignments
        )

        # Run subsampler and store results
        new_results = subsampler(model, data; initial_assignments=assignments)
        assignments = last(new_results.assignments)
        append_results!(results, new_results, S)
    end

    return results
end

"""
Anneals priors
"""
function anneal(priors::NeymanScottPriors, temp::Float64, method::Symbol)

    if method === :cluster_size

        # Start with large variance in cluster amplitudes, which
        # encourages small clusters to form early in sampling.
        clus_amp_mean = mean(priors.cluster_amplitude)
        clus_amp_var = temp * var(priors.cluster_amplitude)

        # Return priors with modified cluster amplitude distribution.
        return NeymanScottPriors(
            priors.cluster_rate,
            specify_gamma(clus_amp_mean, clus_amp_var),
            priors.bkgd_amplitude,
            priors.cluster_priors,
        )

    elseif method === :bkgd

        # Start with low-amplitude background process, which 
        # forces datapoints to be assigned to clusters early on.
        bkgd_amp_mean = (1 / temp) * mean(priors.bkgd_amplitude)
        bkgd_amp_var = var(priors.bkgd_amplitude)

        # Return priors with modified cluster amplitude distribution.
        return NeymanScottPriors(
            priors.cluster_rate,
            priors.cluster_amplitude,
            specify_gamma(bkgd_amp_mean, bkgd_amp_var),
            priors.cluster_priors,
        )

    elseif method === :rate

        # Start with high rate of latent events.
        return NeymanScottPriors(
            temp * priors.cluster_rate,
            priors.cluster_amplitude,
            priors.bkgd_amplitude,
            priors.cluster_priors,
        )

    else
        error("Invalid annealing method specified.")
    end
end
