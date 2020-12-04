struct GibbsSampler <: AbstractSampler
    verbose::Bool
    save_interval::Int
    save_keys::Union{Symbol, Tuple{Vararg{Symbol}}}
    num_samples::Int
end

function GibbsSampler(
    ; verbose=true, 
    save_interval=1, 
    save_keys=(:log_p, :assignments, :events, :globals), 
    num_samples=100
)
    return GibbsSampler(verbose, save_interval, save_keys, num_samples)
end

function (S::GibbsSampler)(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint};
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    
    # Grab sampling options.
    verbose, save_interval, num_samples = S.verbose, S.save_interval, S.num_samples

    # Initialize cluster assignments.
    assignments = initialize_assignments(data, initial_assignments)
    recompute_cluster_statistics!(model, data, assignments)

    # Initialize the globals using a custom function and reset model probabilities
    gibbs_initialize_globals!(model, data, assignments)
    _reset_model_probs!(model)

    results = initialize_results(model, assignments, S)
    data_order = collect(1:length(data))

    for s in 1:num_samples

        # Shuffle order for sampling the assignments.
        shuffle!(data_order)

        # Update cluster assignments
        for i in data_order

            # Remove i-th datapoint from its current cluster.
            if assignments[i] != -1
                remove_datapoint!(model, data[i], assignments[i])
            else
                remove_bkgd_datapoint!(model, data[i])
            end

            # Sample a new assignment for i-th datapoint.
            assignments[i] = gibbs_sample_assignment!(model, data[i])

        end

        # Update cluster parameters.
        for cluster in clusters(model)
            gibbs_sample_cluster_params!(cluster, model)
        end

        # Update global variable.
        gibbs_sample_globals!(model, data, assignments)
        _reset_model_probs!(model)  # TODO -- I think this should be done inside gibbs_sample_globals!

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
function gibbs_sample_assignment!(model::NeymanScottModel, x::AbstractDatapoint)

    # Create log-probability vector to sample assignments.
    #
    #  - We need to sample K + 2 possible assignments. We could assign `x` to
    #    one of the K existing clusters. We could also form a new cluster
    #    (index K + 1), or assign `x` to the background (index K + 2).

    # Shape and rate parameters of gamma prior on latent event amplitude.
    α = cluster_amplitude(model.priors).α
    β = cluster_amplitude(model.priors).β

    K = num_clusters(model)
    
    # Grab vector without allocating new memory.
    log_probs = resize!(model.K_buffer, K + 2)

    # Iterate over clusters, indexed by k = {1, 2, ..., K}.
    for (k, cluster) in enumerate(clusters(model))

        # Check if `cluster` is too far away from `k` to be considered.
        # When performing this check, we need to make sure that the cluster
        # parameters have been sampled --- failing to do this check
        # previously led to a subtle bug, which was very painful to fix.
        if too_far(x, cluster, model) && been_sampled(cluster)
            @debug "Too far!"
            log_probs[k] = -Inf

        # Compute probability of adding x to k-th cluster.
        else
            Nk = datapoint_count(cluster)
            log_probs[k] = log(Nk + α) + log_posterior_predictive(cluster, x, model)
        end
    end
    
    # New cluster probability.
    log_probs[K + 1] = model.new_cluster_log_prob + log_posterior_predictive(x, model)

    # Background probability
    log_probs[K + 2] = model.bkgd_log_prob + bkgd_log_like(model, x)

    # Sample new assignment for x.
    z = sample_logprobs!(log_probs)

    # Assign datapoint to background.
    if z == (K + 2)
        add_bkgd_datapoint!(model, x)
        return -1

    # Assign datapoint to a new cluster. Note that the `add_event!` 
    # function returns a newly allocated assignment index.
    elseif z == (K + 1)
        return add_cluster!(model, x)

    # Assign datapoint to an existing cluster. There are `K` existing
    # clusters with integer ids held in clusters(model).indices ---
    # we use z as an index into this list.
    else
        k = clusters(model).indices[z]
        add_datapoint!(model, x, k)
        return k
    end
end
