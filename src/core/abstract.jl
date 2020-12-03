# ===
# Code for abstract types defining a Neyman-Scott model.
# ===

abstract type AbstractDatapoint{N} end

abstract type AbstractCluster{N} end

abstract type AbstractGlobals end

abstract type AbstractPriors end

abstract type AbstractMask end

struct ClusterList{C <: AbstractCluster}
    clusters::Vector{C}
    indices::Vector{Int64}
end

mutable struct NeymanScottModel{
    N,
    D <: AbstractDatapoint{N},
    E <: AbstractCluster{N},
    G <: AbstractGlobals,
    P <: AbstractPriors
}
    bounds  # Float or tuple
    max_event_radius

    priors::P
    globals::G
    clusters::ClusterList{E}

    new_cluster_log_prob
    bkgd_log_prob

    K_buffer::Vector
    buffers
end


const NOT_SAMPLED_AMPLITUDE = -1.0




"""
Abstract type for observed datapoints.
"""
AbstractDatapoint

position(point::AbstractDatapoint) = point.position

function first_coordinate(point::AbstractDatapoint{N}) where {N}
    return (N > 1) ? position(point)[1] : position(point)
end




"""
Abstract type for clusters (i.e. "latent events").

Subtypes `E` must contain the fields:

    datapoint_count
    sampled_position
    sampled_amplitude

or override the behavior of the associated get methods.
"""
AbstractCluster

datapoint_count(cluster::AbstractCluster) = cluster.datapoint_count

position(cluster::AbstractCluster) = cluster.sampled_position

amplitude(cluster::AbstractCluster) = cluster.sampled_amplitude

been_sampled(e::AbstractCluster) = amplitude(e) > 0

function first_coordinate(cluster::AbstractCluster{N}) where {N}
    return (N > 1) ? position(cluster)[1] : position(cluster)
end




"""
Abstract type for global variables.

Subtypes `G` must contain the fields

    bkgd_rate

or override the behavior of the associated get methods.
"""
AbstractGlobals

bkgd_rate(globals::AbstractGlobals) = globals.bkgd_rate




"""
Abstract type for priors.

Subtypes `P` must contain the fields:

    cluster_rate
    cluster_amplitude
    bkgd_amplitude

or override the behavior of the associated get methods.
"""
AbstractPriors

cluster_rate(priors::AbstractPriors) = priors.cluster_rate

cluster_amplitude(priors::AbstractPriors) = priors.cluster_amplitude

bkgd_amplitude(priors::AbstractPriors) = priors.bkgd_amplitude