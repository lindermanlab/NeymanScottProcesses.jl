# ===
#
# Abstract types
#
# ===

"""
    AbstractCluster

Subtypes are structs that holds sufficient statistics
for all datapoints assigned to a cluster.

## Subtypes must contain the fields:

- `datapoint_count`
- `sampled_position`
- `sampled_amplitude`
"""
abstract type AbstractCluster end

Base.size(cluster::AbstractCluster) = cluster.datapoint_count

been_sampled(cluster::AbstractCluster) = cluster.sampled_amplitude > 0

function add_datapoint!(
    cluster::AbstractCluster,
    datapoint, 
)
    cluster.datapoint_count += 1
    update!(cluster, datapoint)
end

function remove_datapoint!(
    cluster::AbstractCluster,
    datapoint,
)
    cluster.datapoint_count -= 1
    downdate!(cluster, datapoint)
end



"""
    ClusterPriors

Subtypes are structs that hold hyperparameters defining
the prior distribution over cluster parameters.
"""
abstract type ClusterPriors end

"""
    ClusterGlobals

Subtypes are structs that hold any sampled latent variables
that are shared among all clusters.
"""
abstract type ClusterGlobals end


# ===
#
# ClusterList type --- see core/cluster_list.jl
#
# ===

"""
    ClusterList{C <: AbstractCluster}

Holds list of non-empty cluster structs and assignment
indices.

##Fields
clusters :
    Vector of clusters, some may be empty.

indices :
    Sorted vector of unique integer ids, specifying the
    indices of non-empty clusters. Note that
    `length(indices) <= length(clusters)`, with equality if
    and only if there are no empty clusters.
"""
struct ClusterList{C <: AbstractCluster}
    clusters::Vector{C}
    indices::Vector{Int64}
end

function ClusterList(clusters::Vector{C}) where {C <: AbstractCluster}
    return ClusterList{C}(clusters, collect(1:length(clusters)))
end

# ===
#
# Priors for Neyman-Scott model
#
# ===

"""
    NeymanScottPriors{C <: AbstractCluster}

This specifies a homogeneous Poisson process prior over the
background process and a homogeneous Poisson process prior
over the latent events (i.e. clusters).

## Fields
- `cluster_rate::Float64`

Rate of homogeneous Poisson process generating clusters (i.e. latent events).

- `cluster_amplitude::RateGamma`

Gamma distribution specifying prior over the expected size of each cluster.

- `bkgd_rate::RateGamma`

Gamma distribution specifying prior over the background rate.

- `cluster_priors<:ClusterPriors`

Struct holding priors specific to the parametric form of the cluster (e.g., `GaussianClusterPriors`).

- `new_cluster_log_prob::Float64`

Log probability of assigning a singleton parent assignment to a new cluster, `log α + α⋅(log(β) - log(1 + β)`.
"""
struct NeymanScottPriors{C <: AbstractCluster}
    cluster_rate::Float64    
    cluster_amplitude::RateGamma
    bkgd_amplitude::RateGamma
    cluster_priors::ClusterPriors
    new_cluster_log_prob::Float64
end

# Main constructor.
function NeymanScottPriors(
    cluster_rate::Float64,
    cluster_amplitude::RateGamma,
    bkgd_amplitude::RateGamma,
    cluster_priors::ClusterPriors
)
    C = get_cluster_type(cluster_priors)
    α, β = cluster_amplitude.α, cluster_amplitude.β
    new_cluster_log_prob = log(α) + α * (log(β) - log(1 + β))
    return NeymanScottPriors{C}(
        cluster_rate,
        cluster_amplitude,
        bkgd_amplitude,
        cluster_priors,
        new_cluster_log_prob
    )
end

# ===
#
# Global variables for the Neyman-Scott model
#
# ===

"""
    NeymanScottGlobals{C <: AbstractCluster}

This holds global variables associated with the background
process (homogeneous Poisson process over observed events)
and the latent event process (homogeneous Poisson process
over clusters).

## Fields
- `bkgd_rate::Float64`

The background rate parameter. This values changes over successive Gibbs samples.

- `bkgd_log_prob::Float64`

Log probability of assigning an observed event to the background process.
This value changes

- `cluster_globals<:ClusterGlobals`

Struct specifying additional global variables specific to the clustering
model.
"""
mutable struct NeymanScottGlobals{C <: AbstractCluster}
    bkgd_rate::Float64
    bkgd_log_prob::Float64
    cluster_globals::ClusterGlobals
end


# ===
#
# The full model.
#
# ===

"""
    NeymanScottModel{C <: AbstractCluster}

Neyman-Scott point process model.

## Fields
- `domain::Region`

The region of space-time the model is defined over.

- `priors::NeymanScottPriors{C}`

Specifies all prior distributions.

- `globals::NeymanScottGlobals{C}`

Mutable struct holding global variables. These variables
change over the course of MCMC sampling.

- `clusters::ClusterList{C}`

Dynamically re-sized list of clusters.

- `_log_probs_buffer::Vector{Float64}`

Vector used internally by Gibbs sampling routines
(prevents re-allocating memory in inner sampling loops).
"""
mutable struct NeymanScottModel{C <: AbstractCluster}
    domain::Region
    priors::NeymanScottPriors{C}
    globals::NeymanScottGlobals{C}
    cluster_list::ClusterList{C}
    _log_probs_buffer::Vector{Float64}
end

# TODO: consider making this immutable?

function NeymanScottModel(
    domain::Region,
    priors::NeymanScottPriors{C}
) where C <: AbstractCluster

    # Initialize with an empty cluster
    empty_cluster = C(domain)

    # Sample initial global variables from priors.
    globals = sample_globals(domain, priors)

    # Create model.
    return NeymanScottModel(
        domain,
        priors,
        globals,
        ClusterList(empty_cluster),
        Float64[] # buffer for log probabilities
    )
end

"""
Return the type of an observation. For example,

    observations_type(model{GaussianCluster}) -> Vector{Float64}
"""
observations_type(model::NeymanScottModel) = observations_type(model.domain)


# datapoint_count(cluster::AbstractCluster) = cluster.datapoint_count

# position(cluster::AbstractCluster) = cluster.sampled_position

# amplitude(cluster::AbstractCluster) = cluster.sampled_amplitude


# function first_coordinate(cluster::AbstractCluster{N}) where {N}
#     return (N > 1) ? position(cluster)[1] : position(cluster)
# end


# priors(model::NeymanScottModel) = model.priors

# get_priors(model::NeymanScottModel) = priors(model)

# globals(model::NeymanScottModel) = model.globals

# get_globals(model::NeymanScottModel) = globals(model)

# clusters(model::NeymanScottModel) = model.clusters

# labels(model::NeymanScottModel) = labels(clusters(model))

# bounds(model::NeymanScottModel) = model.bounds

# max_cluster_radius(model::NeymanScottModel) = model.max_cluster_radius

# num_clusters(model::NeymanScottModel) = length(clusters(model))

# first_bound(model::NeymanScottModel{N, D, E, P, G}) where {N, D, E, P, G} =
#     (N > 1) ? bounds(model)[1] : bounds(model)

# cluster_rate(priors::NeymanScottPriors) = priors.cluster_rate

# cluster_amplitude(priors::NeymanScottPriors) = priors.cluster_amplitude

# bkgd_amplitude(priors::NeymanScottPriors) = priors.bkgd_amplitude

# bkgd_rate(globals::NeymanScottGlobals) = globals.bkgd_rate


# """Reset new cluster and background probabilities."""
# function _reset_model_probs!(model::NeymanScottModel)
#     P = priors(model)
#     G = globals(model)

#     Ak = cluster_amplitude(P)
#     α, β = Ak.α, Ak.β

#     model.bkgd_log_prob = (
#         log(bkgd_rate(G))
#         + log(volume(model))
#         + log(1 + β)
#     )
#     model.new_cluster_log_prob = (
#         log(α)
#         + log(cluster_rate(P))
#         + log(volume(model))
#         + α * (log(β) - log(1 + β))
#     )
 
# end