struct GibbsSampler <: AbstractSampler
    verbose::Bool
    save_interval::Int
    save_keys::Union{Symbol, Tuple{Vararg{Symbol}}}
    num_samples::Int
end

function GibbsSampler(
    ; verbose=true, 
    save_interval=1, 
    save_keys=(:log_p, :assignments, :clusters, :globals), 
    num_samples=100
)
    return GibbsSampler(verbose, save_interval, save_keys, num_samples)
end

function (S::GibbsSampler)(
    model::NeymanScottModel, 
    data::Vector;
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    
    # Grab sampling options.
    verbose, save_interval, num_samples = S.verbose, S.save_interval, S.num_samples

    # Initialize cluster assignments.
    assignments = initialize_assignments(data, initial_assignments)
    recompute_cluster_statistics!(model, data, assignments)

    # Initialize the globals using a custom function and reset model probabilities
    # gibbs_initialize_globals!(model, data, assignments)
    # _reset_model_probs!(model)
    gibbs_sample_globals!(
        model.globals, model.domain, model.priors, data, assignments
    )

    results = initialize_results(model, assignments, S)
    data_order = collect(1:length(data))

    for s in 1:num_samples

        # Shuffle order for sampling the assignments.
        shuffle!(data_order)

        # Update cluster assignments
        for i in data_order

            # Remove i-th datapoint from its current cluster.
            if assignments[i] != -1
            
                # Get assigned cluster.
                k = assignments[i]
                cluster = model.cluster_list[k]

                # If datapoint i is the last item in cluster, remove the
                # cluster entirely.
                if size(cluster) == 1
                    remove_cluster!(model.cluster_list, k)

                # Otherwise, remove datapoint i from the cluster, and
                # update the sufficient statistics.
                else
                    @show cluster.datapoint_count
                    @show eigvals(cluster.second_moment)
                    remove_datapoint!(cluster, data[i])
                    @show cluster.datapoint_count
                    @show eigvals(cluster.second_moment)
                    recompute_posterior!(cluster, model.priors.cluster_priors)
                end

            # If datapoint i is in the background partition, we need a
            # special function to remove it.
            else
                remove_bkgd_datapoint!(model, data[i])
            end

            # Sample a new assignment for i-th datapoint.
            assignments[i] = gibbs_sample_assignment!(model, data[i])

        end

        # Update cluster parameters.
        for cluster in model.cluster_list
            gibbs_sample_cluster_params!(
                cluster, model.priors
            )
        end

        # Update global variable.
        gibbs_sample_globals!(
            model.globals, model.domain, model.priors, data, assignments
        )

        # Recompute background and new cluster probabilities
        recompute_cluster_statistics!(model, data, assignments)

        # Store results
        if (s % save_interval) == 0
            j = Int(s / save_interval)
            update_results!(results, model, assignments, data, S)
            verbose && print(s, "-")  # Display progress
        end

    end

    verbose && println("Done")

    return results
end

"""
Samples cluster assignment variable for datapoint `x` according to its
conditional distribution, and updates the cluster statistics in `model`
accordingly. Returns integer corresponding the new cluster assignment.
"""
function gibbs_sample_assignment!(model::NeymanScottModel, x)

    # Create log-probability vector to sample assignments.
    #
    #  - We need to sample K + 2 possible assignments. We could assign `x` to
    #    one of the K existing clusters. We could also form a new cluster
    #    (index K + 1), or assign `x` to the background (index K + 2).

    # Shape and rate parameters of gamma prior on latent cluster amplitude.
    α = model.priors.cluster_amplitude.α
    β = model.priors.cluster_amplitude.β

    # Number of non-empty clusters.
    K = length(model.cluster_list.indices)
    
    # Grab vector without allocating new memory.
    log_probs = resize!(model._log_probs_buffer, K + 2)

    # Iterate over clusters.
    #   Note that i = {1, ..., K} are *not* necessarily the cluster
    #   indices, which are contained in cluster.cluster_list.indices.
    for (i, cluster) in enumerate(model.cluster_list)

        # # Check if `cluster` is too far away from `k` to be considered.
        # # When performing this check, we need to make sure that the cluster
        # # parameters have been sampled --- failing to do this check
        # # previously led to a subtle bug, which was very painful to fix.
        # if too_far(x, cluster, model) && been_sampled(cluster)
        #     @debug "Too far!"
        #     log_probs[k] = -Inf
        # # Compute probability of adding x to k-th cluster.
        # else
        #     Nk = datapoint_count(cluster)
        #     log_probs[k] = log(Nk + α) + log_posterior_predictive(cluster, x, model)
        # end

        Nk = cluster.datapoint_count
        log_probs[i] = log(Nk + α) + log_posterior_predictive(cluster, x, model)

    end
    
    # New cluster probability.
    log_probs[K + 1] = model.priors.new_cluster_log_prob + log_posterior_predictive(x, model)

    # Background probability
    log_probs[K + 2] = model.globals.bkgd_log_prob + bkgd_log_like(model, x)

    # Sample new assignment for x.
    z = sample_logprobs!(log_probs)

    # Assign datapoint to background.
    if z == (K + 2)
        add_bkgd_datapoint!(model, x)
        return -1

    # Assign datapoint to a new cluster. Note that the `add_cluster!` 
    # function returns a newly allocated assignment index.
    elseif z == (K + 1)
        k = add_cluster!(model, x)
        recompute_posterior!(
            model.cluster_list[k], model.priors.cluster_priors
        )
        return k

    # Assign datapoint to an existing cluster. There are `K` existing
    # clusters with integer ids held in model.cluster_list.indices, so
    # we use z as an index into this list.
    else
        k = model.cluster_list.indices[z]
        cluster = model.cluster_list[k]
        add_datapoint!(cluster, x)
        recompute_posterior!(
            cluster, model.priors.cluster_priors
        )
        return k
    end
end

function gibbs_sample_globals!(
    globals::NeymanScottGlobals,
    domain::Region,
    priors::NeymanScottPriors,
    data::Vector, 
    assignments::Vector{Int}
)

    # Sample homogeneous background rate.
    globals.bkgd_rate = rand(posterior(
        volume(domain),
        count(==(-1), assignments),
        priors.bkgd_amplitude
    ))

    # Update probability of assignment to background.
    globals.bkgd_log_prob = (
        log(globals.bkgd_rate)
        + log(volume(domain))
        + log(1 + priors.cluster_amplitude.β)
    )

    # Sample any cluster-specific global variables.
    gibbs_sample_globals!(
        globals.cluster_globals,
        priors.cluster_priors,
        data,
        assignments
    )

end


"""
Samples cluster-specific parameters. First samples the latent amplitude
parameter associated with all Neyman-Scott models and then samples
noise-model specific parameters.
"""
function gibbs_sample_cluster_params!(
    cluster::AbstractCluster,
    priors::NeymanScottPriors
)
    n = cluster.datapoint_count
    cluster.sampled_amplitude = rand(posterior(n, priors.cluster_amplitude))
    gibbs_sample_cluster_params!(cluster, priors.cluster_priors)
end

