
"""
Initializes cluster assignments.
"""
function initialize_clusters(
        data::AbstractVector,
        num_clusters::Integer,
        prior::AbstractPrior,
        Cluster::DataType
    )

    # Dimensions of dataset.
    n_pts = length(data)
    num_clusters = min(n_pts, num_clusters)

    # Create random cluster assignments.
    assignments = 1 .+ (collect(0:(n_pts - 1)) .% num_clusters)
    rnd.shuffle!(assignments)

    # Create empty clusters.
    clusters = [Cluster(data[1], prior) for k in 1:num_clusters]

    # Add datapoints to clusters.
    for i in 1:n_pts
        k = assignments[i]
        update_suffstats!(clusters[k], data[i])
    end

    # Add an empty cluster at the end.
    push!(clusters, Cluster(data[1], prior))

    return assignments, clusters
end
