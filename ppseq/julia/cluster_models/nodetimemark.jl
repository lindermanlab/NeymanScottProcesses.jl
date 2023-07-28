
"""
Each datapoint is contains an embassy (encoded
as a one-hot sparse vector). A timestamp (encoded
as float). And a vector of word counts (encoded
as a sparse vector).
"""
const NodeTimeMarkDatapoint = Tuple{
    SparseVector{Int64,Int64},
    Float64,
    SparseVector{Int64,Int64}
}


"""
Stores sufficient statistics for a cables data

_node : sufficient statistics related to embassy identity.
_time : sufficient statistics related to latent event time.
_mark : sufficient statistics related to document features.
"""
mutable struct NodeTimeMarkCluster <: AbstractCluster
    _node::MultinomialCluster
    _time::GaussianCluster
    _mark::MultinomialCluster
    prior::CrossPrior
end


function NodeTimeMarkCluster(
        datapoint::NodeTimeMarkDatapoint,
        prior::CrossPrior
    )

    n, t, m = datapoint
    return NodeTimeMarkCluster(
        MultinomialCluster(n, prior.priors[1]),
        GaussianCluster([t], prior.priors[2]),
        MultinomialCluster(m, prior.priors[3]),
        prior
    )
end


"""
Adds datapoint to cluster.
"""
function update_suffstats!(
        c::NodeTimeMarkCluster,
        datapoint::NodeTimeMarkDatapoint
    )
    update_suffstats!(c._node, datapoint[1])
    update_suffstats!(c._time, [datapoint[2]])
    update_suffstats!(c._mark, datapoint[3])
end


"""
Removes datapoint from cluster.
"""
function downdate_suffstats!(
        c::NodeTimeMarkCluster,
        datapoint::NodeTimeMarkDatapoint
    )
    downdate_suffstats!(c._node, datapoint[1])
    downdate_suffstats!(c._time, [datapoint[2]])
    downdate_suffstats!(c._mark, datapoint[3])
end

# where is posterior being used??

"""
Computes posterior distribution of node / time /mark
"""
posterior(c::NodeTimeMarkCluster) = (
    posterior(c._node),
    posterior(c._time),
    posterior(c._mark),
)
