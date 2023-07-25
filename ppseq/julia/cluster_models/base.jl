"""
Generic type for a cluster of datapoints.
"""
abstract type AbstractCluster end

"""
Returns number of datapoints in a cluster.
"""
size(c::AbstractCluster) = c._size


"""
Returns log likelihood of a cluster.
"""
log_likelihood(c::AbstractCluster) =
    log_normalizer(posterior(c)) - log_normalizer(c._prior)

"""
Computes log probability of a new observation (predictive posterior).
Does not modify the cluster.
"""
function log_p_add(x::AbstractVector, c::AbstractCluster)
    # Compute log probability before adding new observation
    z0 = log_normalizer(posterior(c))

    # Compute log prob after adding new observation
    update_suffstats!(c, x)
    z1 = log_normalizer(posterior(c))
    downdate_suffstats!(c, x)

    return z1 - z0
end

"""
Computes log probability of an existing observation.
Does not modify the cluster.
"""
function log_p_existing(x::AbstractVector, c::AbstractCluster)
    # Compute log probability before removing observation
    z1 = log_normalizer(posterior(c))

    # Compute log prob after removing observation
    downdate_suffstats!(c, x)
    z0 = log_normalizer(posterior(c))
    update_suffstats!(c, x)

    return z1 - z0
end
