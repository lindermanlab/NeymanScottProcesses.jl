
"""
Create GaussianDPMM and sample via collapsed Gibbs.
"""
function gibbs_sample(
        model::AbstractMixtureModel,
        data::AbstractVector,
        num_samples::Int
    ) where {T<:Real}

    # Pre-allocate storage for cluster assignment samples.
    num_datapoints = length(data)
    z = zeros(Int, num_datapoints, num_samples + 1)
    log_like_hist = zeros(num_samples + 1)
    num_cluster_hist = zeros(Int64, num_samples + 1)

    z[:, 1] = assignments(model)
    log_like_hist[1] = log_likelihood(model)
    num_cluster_hist[1] = length(model.clusters) - 1

    # Draw samples.
    for s = 1:num_samples

        # Run Gibbs scan over all datapoints.
        for i = 1:num_datapoints

            # Compute likelihood of assigning data[:, i] to each
            # cluster (or forming a new cluster).
            logprobs = assign_logprobs(data[i], model, model.assignments[i])

            # Sample new cluster assignment.
            k = sample_logprobs(logprobs)

            # Remove datapoint from its current cluster and add to cluster k
            move_datapoint!(model, data[i], i, k)
        end

        #
        #
        # TODO -- resample datapoints in unobserved regions here.
        #
        #

        # Store current cluster assignments.
        z[:, s + 1] = assignments(model)

        # Store model likelihood and other statistics of interest.
        log_like_hist[s + 1] = log_likelihood(model)
        num_cluster_hist[s + 1] = length(model.clusters) - 1
    end

    return z, model, log_like_hist, num_cluster_hist
end

