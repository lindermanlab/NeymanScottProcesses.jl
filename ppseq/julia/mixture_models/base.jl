"""
Generic type for a mixture model.
"""
abstract type AbstractMixtureModel{P<:AbstractPrior,C<:AbstractCluster} end

# === Methods to Access Model State === #

"""Returns current cluster assignments."""
assignments(model::AbstractMixtureModel) =
    model.assignments

"""Returns cluster sizes. An empty cluster is included at the end."""
cluster_sizes(model::AbstractMixtureModel) =
    [size(c) for c in model.clusters]

"""
Returns log-likelihood of model with current assignments.

TODO: we're missing a term that depends on the number of clusters.
"""
log_likelihood(model::AbstractMixtureModel) =
    sum(log_likelihood(c) for c in model.clusters)


# === Methods to Update Cluster Sufficient Statistics === #

"""
Moves datapoint x, with index i, into cluster k. Adds a new cluster
to the model if necessary. Updates cluster sufficient statistics.
"""
function move_datapoint!(
        model::AbstractMixtureModel{P,C},
        x::Vector{D},
        i::Int64,
        k::Int64,
    ) where {P<:AbstractPrior,C<:AbstractCluster,D<:Real}

    # Old cluster assignment
    old_k = model.assignments[i]
    
    # Skip if new assignment matches old assignment.
    if !(k == old_k)

        # Remove datapoint from old cluster
        downdate_suffstats!(model.clusters[old_k], x)

        # Add x to cluster k.
        model.assignments[i] = k
        update_suffstats!(model.clusters[k], x)

        # If x was added to the final cluster, add a new cluster
        # so that an empty cluster is always retained.
        if k == length(model.clusters)
            new_cluster = C(x, model.cluster_params_prior)
            push!(model.clusters, new_cluster)
        end

        # If x was removed from a singleton cluster, delete the
        # (now empty) cluster and relabel the assignments.
        if size(model.clusters[old_k]) == 0
            
            # Remove cluster object.
            deleteat!(model.clusters, old_k)

            # Relabel assignments.
            for j = 1:length(model.assignments)
                if model.assignments[j] > old_k
                    model.assignments[j] -= 1
                end
            end
        end
    end
end


# === Methods to compute assignment probabilities === #

"""
Computes (unnormalized) log probabilities of assigning a datapoint
x to each existing cluster, and to forming a new cluster (last element).
"""
function assign_logprobs(
        x::Vector{D},
        model::AbstractMixtureModel,
        parent::Integer,
    ) where {D<:Real}

    log_probs = cluster_log_prior(model)

    # Add contribution of predictive posterior.
    for (k, cluster) in enumerate(model.clusters)
        
        # Compute probability of removing x from cluster.
        if k == parent
            log_probs[k] += log_p_existing(x, cluster)

        # Compute probability of adding x to cluster.
        else
            log_probs[k] += log_p_add(x, cluster)

        end
    end

    return log_probs
end
