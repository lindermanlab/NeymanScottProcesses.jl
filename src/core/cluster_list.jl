# ==
# Functionality for ClusterList
# == 

function ClusterList(cluster::C) where C <: AbstractCluster
    ClusterList([cluster], Int64[])
end

"""
    getindex(cluster_list::ClusterList, index::Int)

Given a cluster assignment index, return the associated
cluster struct. Throws an error if index is not
recognized.
"""
function Base.getindex(c::ClusterList, i::Int)

    if !(i in c.indices)
        throw(AssertionError("Tried to access empty cluster."))
    end

    return c.clusters[searchsortedfirst(c.indices, i)]

end

Base.length(c::ClusterList) = length(c.indices)

Base.iterate(c::ClusterList) = (
    isempty(c.indices) ? nothing : (c.clusters[c.indices[1]], 2)
)

Base.iterate(c::ClusterList, i::Int64) = (
    (i > length(c)) ? nothing : (c.clusters[c.indices[i]], i + 1)
)


"""
Create a singleton cluster containing datapoint `x` and return new 
assignment index `k`.
"""
function add_cluster!(model::NeymanScottModel, x)
    
    # Ask cluster list to provide us with an empty cluster.
    k = add_cluster!(model.cluster_list, model.domain)

    # Grab cluster at index k, and add the datapoint to it.
    cluster = model.cluster_list[k]
    add_datapoint!(cluster, x)

    # Return the cluster assignment for datapoint x.
    return k
end


"""
Adds a new cluster. This function first checks if `clusters` contains
an already initialized, but empty, cluster. Otherwise a new empty
cluster is initialized and used.

The integer id of the new cluster (used for assignments) is returned.
"""
function add_cluster!(
        cluster_list::ClusterList{C},
        domain::Region
    ) where C <: AbstractCluster

    # Check if any indices are skipped. If so, use the smallest skipped
    # integer as the index for the new cluster.
    i = 1
    for j in cluster_list.indices

        # We have j == c.indices[i].

        # If (j == indices[i] != i) then the i-th cluster is empty.
        if i != j
            # Mark i-th cluster as no longer empty. Then return
            # i as the cluster id.
            insert!(cluster_list.indices, i, i)
            return i
        end

        # Increment to check if (i + 1)-th cluster is empty.
        i += 1

    end

    # If we reached here without returning, then indices is a vector 
    # [1, 2, ..., K] without any skipped integers. So we'll use K + 1
    # as the new integer index.
    push!(
        cluster_list.indices,
        length(cluster_list.indices) + 1
    )

    # If needed, create and append an empty cluster by calling the
    # constructor for `C` (where C <: AbstractCluster).
    if length(cluster_list.clusters) < length(cluster_list.indices)
        push!(cluster_list.clusters, C(domain))
    end

    # Return index of the empty Cluster struct.
    return cluster_list.indices[end]
end


"""
Marks a Cluster struct as empty and resets its sufficient statistics. This does not delete 
the Cluster.
"""
function remove_cluster!(cluster_list::ClusterList, index::Int64)
    empty!(cluster_list.clusters[index])
    return deleteat!(
        cluster_list.indices,
        searchsorted(cluster_list.indices, index)
    )
end


"""
Recompute cluster sufficient statistics.
"""
function recompute_cluster_statistics!(
    model::NeymanScottModel{C},
    datapoints::AbstractVector,
    assignments::AbstractVector{Int64}
) where C <: AbstractCluster
    
    # Grab clusters
    cluster_list = model.cluster_list

    # Reset all clusters to empty.
    for k in cluster_list.indices
        empty!(cluster_list.clusters[k])
    end
    empty!(cluster_list.indices)

    # Add datapoints back to their previously assigned cluster.
    for (x, k) in zip(datapoints, assignments)
        
        # Skip datapoints assigned to the background.
        (k < 0) && continue

        # Ensure that cluster k exists (even if empty).
        while k > length(cluster_list.clusters)
            push!(cluster_list.clusters, C(model.domain))
        end

        # Mark cluster k as not empty.
        j = searchsortedfirst(cluster_list.indices, k)
        if (j > length(cluster_list.indices)) || (cluster_list.indices[j] != k)
            insert!(cluster_list.indices, j, k)
        end

        # Add datapoint x to k-th cluster.
        add_datapoint!(model.cluster_list[k], x)
    end

    # Now that all sufficient statistics are computed, we can 
    # compute the statistics defining the posterior predictive distribution.
    for cluster in cluster_list
        recompute_posterior!(cluster, model.priors.cluster_priors)
    end
end
