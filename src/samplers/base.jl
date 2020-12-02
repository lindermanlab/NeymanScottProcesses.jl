"""
    AbstractSampler

An abstract base type for samplers. Subtypes `S::Sampler` must implement the following 
properties

    S.verbose
    S.save_interval
    S.save_keys
    S.num_samples

And the following methods

    (optional) valid_save_keys(S::Sampler)
    (S::Sampler)(
        model::NeymanScottModel, 
        data::Vector{<: AbstractDatapoint};
        initial_assignments::Union{Symbol, Vector{Int64}}=:background
    )
"""
abstract type AbstractSampler end

valid_save_keys(::AbstractSampler) = (:log_p, :assignments, :events, :globals)




# ===
# SAVING
# ===

_dictkeys(d::Dict) = (collect(keys(d))...,)
_dictvalues(d::Dict) = (collect(values(d))...,)
_namedtuple(d::Dict) = NamedTuple{_dictkeys(d)}(_dictvalues(d))

"""
Initialize sampler results.
"""
function initialize_results(model, assignments, S::AbstractSampler)
    save_interval, save_keys, num_samples = S.save_interval, S.save_keys, S.num_samples

    @assert (:assignments in save_keys)

    n_saved_samples = Int(round(num_samples / save_interval))
    if save_keys === :all
        save_keys = valid_save_keys(S)
    end

    results = Dict{Symbol, Any}()
    for key in save_keys
        @assert key in valid_save_keys(S)
        results[key] = []
    end

    return _namedtuple(results)
end

"""
Update sampler results.
"""
function update_results!(results, model, assignments, data, S::AbstractSampler)
    save_keys = S.save_keys

    if save_keys == :all
        save_keys = valid_save_keys(S)
    end

    if :log_p in save_keys
        push!(results[:log_p], log_like(model, data))
    end

    if :assignments in save_keys
        push!(results[:assignments], deepcopy(assignments))
    end

    if :latents in save_keys
        push!(results[:events], deepcopy(event_list_summary(model)))
    end

    if :globals in save_keys
        push!(results[:globals], deepcopy(get_globals(model)))
    end

    return results
end

"""
Append results from a subsampler to its parent's results.
"""
function append_results!(results, new_results, S::AbstractSampler)
    save_keys = S.save_keys

    if save_keys == :all
        save_keys = valid_save_keys(S)
    end

    for key in save_keys
        append!(results[key], new_results[key])
    end

    return results
end




# ===
# INITIALIZATION
# ===

"""
Initialize assignments. If `:background` was passed, initialize all assignments to the
background process (label `-1`). Otherwise, make a copy of the passed initial assignments.
"""
function initialize_assignments(data, initial_assignments)
    if initial_assignments === :background
        initial_assignments = fill(-1, length(data))
    end

    # Initialize spike assignments.
    return deepcopy(initial_assignments)
end
