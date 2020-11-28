struct Annealer <: AbstractSampler
    verbose::Bool
    temperatures::Vector{Real}
    anneal_fn::Union{Function, Symbol}
    subsampler::AbstractSampler
end

valid_save_keys(S::Annealer) = valid_save_keys(S.subsampler)

function Annealer(
    sampler::AbstractSampler, max_temp::Real, anneal_fn::Union{Function, Symbol}; 
    verbose=true, num_anneals=10
)
    temps = exp10.(range(log10(max_temperature), 0, length=num_samples))
    return Annealer(verbose, save_keys, temps, anneal_fn, subsampler)
end

function Base.getproperty(obj::Annealer, sym::Symbol)
    if sym === :save_interval
        return 1
    elseif sym === :num_samples
        return length(getfield(obj, :temperatures))
    elseif sym === :save_keys
        return getfield(obj, subsampler).save_keys
    else
        return getfield(obj, sym)
    end
end

"""
Convert symbols into anneal functions.
"""
function get_anneal_function(anneal_fn::Union{Function, Symbol}, priors)
    if typeof(anneal_fn) <: Function
        return anneal_fn

    elseif anneal_fn in propertynames(priors)
        function f(true_priors, T)
            new_priors = deepcopy(true_priors)
            val = T * getproperty(true_priors, anneal_fn)
            setproperty!(new_priors, anneal_fn, val)
            return new_priors
        end
        return f
    
    else
        error(
            "Invalid annealing function." *
            "Either pass a function f(priors, temp) or a symbol in the priors."
        )
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
    verbose, temperatures, anneal_fn, sampler = S.verbose, S.temperatures, S.anneal_fn, S.sampler
    
    # Set up annealing function
    # TODO Should we verify the annealing function is valid?
    true_priors = deepcopy(priors)
    anneal_fn = get_anneal_function(anneal_fn, model.priors)
    
    # Initialize assignments and results
    assignments = initialize_assignments(data, initial_assignments)
    results = initialize_results(model, assignments, S)

    for temp in temperatures
        verbose && println("TEMP:  ", temp)

        # Anneal and recompute restaurant process probabilities
        model.priors = anneal_fn(true_priors, temp)
        _reset_model_probs!(model)

        new_results = sampler(model, data, assignments)
        append_results!(results, new_results)
    end

    return results
end
