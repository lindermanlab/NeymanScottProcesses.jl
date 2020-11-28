struct Annealer <: AbstractSampler
    verbose::Bool
    temperatures::Vector{Real}
    anneal_fn::Union{Function, Symbol}
    subsampler::AbstractSampler
end

valid_save_keys(S::Annealer) = valid_save_keys(S.subsampler)

function Annealer(
    subsampler::AbstractSampler, max_temp::Real, anneal_fn::Union{Function, Symbol}; 
    verbose=true, num_samples=10
)
    temps = exp10.(range(log10(max_temp), 0, length=num_samples))
    return Annealer(verbose, temps, anneal_fn, subsampler)
end

function Base.getproperty(obj::Annealer, sym::Symbol)
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
function (S::Annealer)(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint};
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    verbose, temperatures = S.verbose, S.temperatures
    anneal_fn, subsampler = S.anneal_fn, S.subsampler

    # Set up annealing function
    # TODO Alex, should we verify the annealing function is valid?
    true_priors = deepcopy(model.priors)
    anneal_fn = get_anneal_function(anneal_fn, model.priors)
    
    # Initialize assignments and results
    assignments = initialize_assignments(data, initial_assignments)
    results = initialize_results(model, assignments, S)

    for temp in temperatures
        verbose && println("TEMP:  ", temp)

        # Anneal and recompute restaurant process probabilities
        new_priors = deepcopy(true_priors)
        model.priors = anneal_fn(new_priors, temp)
        _reset_model_probs!(model)

        # Run subsampler and store results
        new_results = subsampler(model, data; initial_assignments=assignments)
        assignments = last(new_results.assignments)
        append_results!(results, new_results, S)
    end

    return results
end

"""
Convert symbols into anneal functions.
"""
function get_anneal_function(anneal_fn::Union{Function, Symbol}, priors)
    if typeof(anneal_fn) <: Function
        f = anneal_fn

    elseif anneal_fn === :event_amplitude_var
        f = function (priors::AbstractPriors, T)
            new_mean = mean(priors.event_amplitude)
            new_var = T * var(priors.event_amplitude)
            priors.event_amplitude = specify_gamma(new_mean, new_var)
            return priors
        end

    elseif anneal_fn === :background_amplitude_mean
        f = function (priors::AbstractPriors, T)
            new_mean = (1/T) * mean(priors.bkgd_amplitude)
            new_var = var(priors.bkgd_amplitude)
            priors.bkgd_amplitude = specify_gamma(new_mean, new_var)
            return priors
        end

    elseif anneal_fn in propertynames(priors)
        f = function (priors::AbstractPriors, T)
            val = T * getproperty(priors, anneal_fn)
            setproperty!(priors, anneal_fn, val)
            return priors
        end
    
    else
        error(
            "Invalid annealing function."
            * "Either pass a function f(priors, temp) or a `:symbol` representing a"
            * "property in the priors."
        )
    end

    return f
end