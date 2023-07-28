"""
Stores sufficient statistics for a cluster of n datapoints, each
described as a d-dimensional vector of count variables, that is
xᵢ ∈ {0, 1, 2, ...} for i ∈ {1, 2, ... , d}.

_size (int) : Number of datapoints assigned to the cluster.
_counts (vector) : Number of observed counts for all d features.
_prior : Dirichlet prior on cluster parameters.

"""
mutable struct MultinomialCluster <: AbstractCluster
    _size::Int64
    _counts::AbstractVector{Int64}
    _prior::DirichletPrior
end


"""
Creates an empty cluster with `n` features.
"""
MultinomialCluster(datapoint::Vector{Int64}, prior::DirichletPrior) = 
    MultinomialCluster(
        0,
        zeros(Int64, length(datapoint)),
        prior
    )

MultinomialCluster(datapoint::SparseVector{Int64, Int64}, prior::DirichletPrior) = 
    MultinomialCluster(
        0,
        spzeros(Int64, length(datapoint)),
        prior
    )


"""
Adds datapoint to cluster.
"""
function update_suffstats!(
        c::MultinomialCluster,
        x::AbstractVector{Int64}
    )
    c._size += 1
    c._counts += x
end

function update_suffstats!(
        c::MultinomialCluster,
        x::Int64
    )
    c._size += 1
    c._counts[x] += 1
end



"""
Removes datapoint from cluster.
"""
function downdate_suffstats!(
        c::MultinomialCluster,
        x::Vector{Int64}
    )
    c._size -= 1
    c._counts -= x
end

function downdate_suffstats!(
        c::MultinomialCluster,
        x::Int64
    )
    c._size -= 1
    c._counts[x] -= 1
end


"""
Posterior distribution of multinomial likelihood and
Dirichlet prior.
"""
function posterior(c::MultinomialCluster)

    # For empty clusters, return the prior distribution.
    if size(c) == 0
        return c._prior
    
    # For non-empty clusters add observed counts to
    # prior pseudocounts.
    else
        return DirichletPrior(c._prior.alpha .+ c._counts)
    end

end
