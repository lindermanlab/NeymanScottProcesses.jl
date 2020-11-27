abstract type AbstractSampler end

is_verbose(S::AbstractSampler) = S.verbose

get_save_interval(S::AbstractSampler) = S.save_interval

get_save_set(S::AbstractSampler) = S.save_set

get_num_samples(S::AbstractSampler) = S.num_samples

valid_save_set(::AbstractSampler) = (:log_p, :assignments, :events, :globals)




# ===
# SAVING
# ===

_dictkeys(d::Dict) = (collect(keys(d))...,)
_dictvalues(d::Dict) = (collect(values(d))...,)
_namedtuple(d::Dict{Symbol,T}) where {T} = NamedTuple{_dictkeys(d)}(_dictvalues(d))

"""Initialize sampler results."""
function initialize_results(model, assignments, S::AbstractSampler)
    save_interval, save_set, num_samples = S.save_interval, S.save_set, S.num_samples

    n_saved_samples = Int(round(num_samples / save_interval))
    if save_set === :all
        save_set = valid_save_set(S)
    end

    results = Dict{Symbol, Any}()
    for key in save_set
        @assert key in valid_save_set(S)
        results[key] = []
    end

    return _namedtuple(results)
end

"""Update sampler results."""
function update_results!(results, model, assignments, data, S::AbstractSampler)
    save_set = S.save_set

    if save_set == :all
        save_set = valid_save_set(S)
    end

    if :log_p in save_set
        push!(results[:log_p], log_like(model, data))
    end

    if :assignments in save_set
        push!(results[:assignments], deepcopy(assignments))
    end

    if :latents in save_set
        push!(results[:events], deepcopy(event_list_summary(model)))
    end

    if :globals in save_set
        push!(results[:globals], deepcopy(get_globals(model)))
    end

    return results
end
