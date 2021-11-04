"""
Dynamically re-sized array holding AbstractCluster structs.

clusters :
    Vector of clusters, some may be empty.

indices :
    Sorted vector of unique integer ids, specifying the
    indices of non-empty clusters. Note that
    `length(indices) <= length(clusters)`, with equality if
    and only if there are no empty clusters.
"""
ClusterList

function ClusterList(cluster::C) where C <: AbstractCluster
    return ClusterList([cluster], Int64[])
end

# labels(ev::ClusterList) = ev.indices

constructor_args(ev::ClusterList) = constructor_args(ev.clusters[1])

Base.getindex(ev::ClusterList, i::Int64) = ev.clusters[i]

Base.length(ev::ClusterList) = length(ev.indices)

Base.iterate(ev::ClusterList) = (
    isempty(ev.indices) ? nothing : (ev.clusters[ev.indices[1]], 2)
)

Base.iterate(ev::ClusterList, i::Int64) = (
    (i > length(ev)) ? nothing : (ev.clusters[ev.indices[i]], i + 1)
)

"""
Adds a new cluster. This function first checks if `clusters` contains
an already initialized, but empty, cluster. Otherwise a new empty
cluster is initialized and used.

The integer id of the new cluster (used for assignments) is returned.
"""
function add_cluster!(cluster_list::ClusterList{C}) where C <: AbstractCluster

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
        push!(cluster_list.clusters, C(constructor_args(cluster_list)...))
    end

    # Return index of the empty Cluster struct.
    return cluster_list.indices[end]
end


"""
Marks a Cluster struct as empty and resets its sufficient statistics. This does not delete 
the Cluster.
"""
function remove_cluster!(cluster_list::ClusterList, index::Int64)
    reset!(cluster_list.clusters[index])
    return deleteat!(
        cluster_list.indices,
        searchsorted(cluster_list.indices, index)
    )
end


"""
Recompute cluster sufficient statistics.
"""
function recompute_cluster_statistics!(
    model::NeymanScottModel,
    cluster_list::ClusterList{C},
    datapoints::Vector{<: AbstractDatapoint},
    assignments::AbstractVector{Int64}
) where C <: AbstractCluster
    

    # Reset all clusters to empty.
    for k in cluster_list.indices
        reset!(cluster_list.clusters[k])
    end
    empty!(cluster_list.indices)

    # Add datapoints back to their previously assigned cluster.
    for (x, k) in zip(datapoints, assignments)
        
        # Skip datapoints assigned to the background.
        (k < 0) && continue

        # Check that cluster k exists.
        while k > length(cluster_list.clusters)
            push!(cluster_list.clusters, C(cluster_args()))
        end

        # Add datapoint x to k-th cluster.
        add_datapoint!(model, x, k, recompute_posterior=false)

        # Make sure that cluster k is marked as non-empty.
        j = searchsortedfirst(cluster_list.indices, k)
        if (j > length(cluster_list.indices)) || (cluster_list.indices[j] != k)
            insert!(cluster_list.indices, j, k)
        end
    end

    # Set the posterior, since we didn't do so when adding datapoints
    for k in cluster_list.indices
        set_posterior!(model, k)
    end
end


function recompute_cluster_statistics_in_place!(
    model::NeymanScottModel,
    cluster_list::ClusterList{C},
    datapoints::Vector{<: AbstractDatapoint},
    assignments::AbstractVector{Int64}
) where C <: AbstractCluster

    # Reset all clusters to empty.
    for k in cluster_list.indices
        reset!(cluster_list.clusters[k])
    end

    # Add datapoints back to their previously assigned cluster.
    for (x, k) in zip(datapoints, assignments)
        
        # Skip datapoints assigned to the background.
        (k < 0) && continue

        # Add datapoint x to k-th cluster.
        add_datapoint!(model, x, k, recompute_posterior=false)
    end

    # Update the posterior, since we didn't do so when adding datapoints
    for k in cluster_list.indices
        set_posterior!(model, k)
    end
end
