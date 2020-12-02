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
    data::Vector{<: AbstractDatapoint},
    masks::Vector{<: AbstractMask}
)
    masked_data = []
    unmasked_data = []

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
Compute log-likelihood in masked regions.
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



# ===
# SAMPLING
# ===

"""
Impute missing data.
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
Impute missing data.
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
