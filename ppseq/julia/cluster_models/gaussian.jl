# =========================================== #
# ==== Multivariate Normal Cluster Model ==== #
# =========================================== #

"""
Stores sufficient statistics for a cluster of datapoints in
ℜⁿ modeled as a multivariate Gaussian.

_size (int) : Number of datapoints assigned to the cluster.
_fm (vector) : Unnormalized first moments of datapoints.
_sm (matrix) : Unnormalized second moments of datapoints.
_prior : Normal-Inverse-Wishart prior on cluster parameters.

"""
mutable struct GaussianCluster <: AbstractCluster
    _size::Int64
    _fm::Vector{Float64}
    _sm::Matrix{Float64}  # TODO: rank-one updates.
    _prior::NormInvWishartPrior
end


"""
Creates an empty cluster with `n` features.
"""
function GaussianCluster(
        datapoint::Vector{Float64},
        prior::NormInvWishartPrior
    )
    n = length(datapoint)  # number of features
    return GaussianCluster(0, zeros(n), zeros(n, n), prior)
end


"""
Adds datapoint to cluster.
"""
function update_suffstats!(
        c::GaussianCluster,
        x::Vector{Float64}
    )
    c._size += 1
    c._fm += x
    c._sm += x * x'  # TODO: rank-one updates.
end


"""
Removes datapoint from cluster.
"""
function downdate_suffstats!(
        c::GaussianCluster,
        x::Vector{Float64}
    )
    c._size -= 1
    c._fm -= x
    c._sm -= x * x'  # TODO: rank-one updates.
end


"""
Computes posterior distribution of cluster mean/covariance.
"""
function posterior(c::GaussianCluster)

    # For empty clusters, return the prior distribution.
    if size(c) == 0
        return c._prior
    end

    # Collect prior and first/second moments.
    prior = c._prior
    fm = c._fm
    sm = c._sm

    # Update number of pseudo-obsevations
    m_n = prior.m_n + size(c)
    S_n = prior.S_n + size(c)

    # Weighted average of prior and empirical mean.
    m = (prior.m_n * prior.m + fm) / m_n

    # Weighted average of prior and empirical covariance.
    S = prior.S + sm + prior.m_n .* (prior.m * prior.m') - m_n .* (m * m')

    # New posterior distribution
    return NormInvWishartPrior(m, m_n, S, S_n)

end
