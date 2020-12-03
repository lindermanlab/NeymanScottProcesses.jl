"""
Abstract type for masks.
"""
AbstractMask

"""
Return true if `x ∈ m` for some `m ∈ masks`.
"""
Base.in(x::AbstractDatapoint, masks::Vector{<: AbstractMask}) = any(m -> (x ∈ m), masks)

"""
Compute percent of model that is masked by `masks`. Assumes disjoint masks.
"""
masked_proportion(model::NeymanScottModel, masks::Vector{<: AbstractMask}) =
    sum(volume.(masks)) / volume(model)

"""
Return two arrays `masked_data, unmasked_data` containing the masked and unmasked 
datapoints, respectively.
"""
function split_data_by_mask(
    data::Vector{D},
    masks::Vector{<: AbstractMask}
) where {D <: AbstractDatapoint}
    masked_data = D[]
    unmasked_data = D[]

    for x in data
        if x in masks
            push!(masked_data, x)
        else
            push!(unmasked_data, x)
        end
    end

    return masked_data, unmasked_data
end




# ===
# LIKELIHOODS
# ===

"""
Compute log-likelihood of masked `data` under sampled parameters in `model`. Every
event held in `data` is assumed to be in the union of all `masks`.
"""
function log_like(
    model::NeymanScottModel,
    data::Vector{<: AbstractDatapoint},
    masks::Vector{<: AbstractMask}
)
    ll = 0.0

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all datapoints -- #
    for x in data
        ll += log_bkgd_intensity(model, x)
        for event in events(model)
            logaddexp(ll, log_event_intensity(model, event, x))
        end
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #
    for mask in masks        
        ll -= integrated_bkgd_intensity(model, mask)
        for event in events(model)
            ll -= integrated_event_intensity(model, event, mask)
        end
    end

    return ll
end

"""
Compute log-likelihood of a homogeneous poisson process of data within a masked region.
"""
function _homogeneous_baseline_log_like(
    data::Vector{<: AbstractDatapoint}, 
    masks::Vector{<: AbstractMask}
)
    return length(data) / sum(volume.(masks))
end

"""
Integrate the intensity of an event in the masked region.
"""
function _integrated_event_intensity(
    model::NeymanScottModel, event::AbstractEvent, mask::AbstractMask;
    num_samples = 1000
)
    num_in_mask = count(i -> (sample_datapoint(event, model) ∈ mask), 1:num_samples)
    prob_in_mask = sum(num_in_mask) / num_samples
    return prob_in_mask * amplitude(event)
end




# ===
# SAMPLING
# ===

"""
Impute missing data by drawing samples from `model`. Samples that fall inside
the censored region, defined by `masks`, are returned. Samples that are not in
the masked region are rejected.
"""
function sample_masked_data(
    model::NeymanScottModel{N, D, E, P, G}, 
    masks::Vector{<: AbstractMask}
) where {N, D, E, P, G}
    data = D[]
    assgn = Int[]

    return sample_masked_data!(data, assgn, model, masks)
end

"""
Overwrite 'data' and `assignments` with new samples from `model`. Samples that fall
inside the censored region, defined by `masks`, are returned. Samples that are not in
the masked region are rejected.
"""
function sample_masked_data!(
    data::Vector{<: AbstractDatapoint},
    assignments::Vector{Int64},
    model::NeymanScottModel,
    masks::Vector{<: AbstractMask}
)
    empty!(data)
    empty!(assignments)

    # Sample background data
    for x in sample_background(globals(model), model)
        if x in masks
            push!(data, x)
            push!(assignments, -1)
        end
    end

    # Sample event data
    for (k, event) in enumerate(events(model))
        z = events(model).indices[k]
        for x in sample(event, globals(model), model)
            if x in masks
                push!(data, x)
                push!(assignments, z)
            end
        end
    end

    return data, assignments
end
