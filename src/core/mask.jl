"""
Abstract type for masks.
"""
AbstractMask

"""
Mask defined by a union over a list of masks.
"""
struct MaskCollection{M <: AbstractMask} <: AbstractMask
    mask_list::Vector{M}
end

Base.in(x::AbstractDatapoint, m::MaskCollection) = any(mask -> (x ∈ mask), m)
Base.length(m::MaskCollection) = length(m.mask_list)
Base.iterate(m::MaskCollection) = iterate(m.mask_list)
Base.iterate(m::MaskCollection, state) = iterate(m.mask_list, state)

# TODO -- this assumes that the masks are not overlapping... Can this be relaxed?
volume(m::MaskCollection) = sum(volume(mask) for mask in m)

"""
Mask defined as the complement of any mask.
"""
struct ComplementMask{M <: AbstractMask} <: AbstractMask
    complement::M
    total_volume::Float64
end

Base.in(x::AbstractDatapoint, m::ComplementMask) = !(x in m.complement)
volume(m::ComplementMask) = m.total_volume - volume(m.complement)

"""
    data_inside_mask, data_outside_mask = split_data_by_mask(data, mask)

Split `data` into two non-overlapping subsets, based on
membership in the region defined by `mask`.
"""
function split_data_by_mask(
    data::Vector{D},
    mask::AbstractMask
) where {D <: AbstractDatapoint}

    data_inside_mask = D[]
    data_outside_mask = D[]

    for x in data
        if x in mask
            push!(data_inside_mask, x)
        else
            push!(data_outside_mask, x)
        end
    end

    return data_inside_mask, data_outside_mask
end




# ===
# LIKELIHOODS
# ===

"""
Compute log-likelihood of masked `data` under sampled parameters in `model`. Every
datapoint in `data` is assumed to be in the union of all `masks`.
"""
function log_like(
    model::NeymanScottModel,
    data::Vector{<: AbstractDatapoint},
    mask::AbstractMask
)
    ll = 0.0

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all datapoints -- #
    for x in data
        ll += log_bkgd_intensity(model, x)
        for cluster in clusters(model)
            logaddexp(ll, log_cluster_intensity(model, cluster, x))
        end
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #
    ll -= integrated_bkgd_intensity(model, mask)
    for cluster in clusters(model)
        ll -= integrated_cluster_intensity(model, cluster, mask)
    end

    return ll
end

"""
Compute log-likelihood of a homogeneous poisson process of data within a masked region.
"""
function _homogeneous_baseline_log_like(
    data::Vector{<: AbstractDatapoint}, 
    mask::AbstractMask
)
    return length(data) / volume(mask)
end

"""
Integrate the intensity of an cluster in the masked region.
"""
function _integrated_cluster_intensity(
    model::NeymanScottModel,
    cluster::AbstractCluster,
    mask::AbstractMask;
    num_samples = 1000
)
    num_in_mask = count(i -> (sample_datapoint(cluster, model) ∈ mask), 1:num_samples)
    prob_in_mask = sum(num_in_mask) / num_samples
    return prob_in_mask * amplitude(cluster)
end




# ===
# SAMPLING
# ===

"""
Impute missing data by drawing samples from `model`. Samples that fall inside
the censored region, defined by `masks`, are returned. Samples that are not in
the masked region are rejected.
"""
function sample_data_in_mask(
    model::NeymanScottModel{N, D, E, P, G}, 
    masks::AbstractMask
) where {N, D, E, P, G}
    data = D[]
    assgn = Int[]

    return sample_data_in_mask!(data, assgn, model, masks)
end

"""
Overwrite 'data' and `assignments` with new samples from `model`. Samples that fall
inside the censored region, defined by `masks`, are returned. Samples that are not in
the masked region are rejected.
"""
function sample_data_in_mask!(
    data::Vector{<: AbstractDatapoint},
    assignments::Vector{Int64},
    model::NeymanScottModel,
    mask::AbstractMask
)
    empty!(data)
    empty!(assignments)

    # Sample background data
    for x in sample_background(globals(model), model)
        if x in mask
            push!(data, x)
            push!(assignments, -1)
        end
    end

    # Sample cluster-evoked datapoints
    for (k, cluster) in enumerate(clusters(model))
        z = clusters(model).indices[k]
        for x in sample(cluster, globals(model), model)
            if x in mask
                push!(data, x)
                push!(assignments, z)
            end
        end
    end

    return data, assignments
end
