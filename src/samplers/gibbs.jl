struct GibbsSampler <: AbstractSampler
    verbose::Bool
    save_interval::Int
    save_set::Union{Symbol, Tuple{Vararg{Symbol}}}
    num_samples::Int
end

GibbsSampler(; verbose=true, save_interval=1, save_set=:all, num_samples=100) = 
    GibbsSampler(verbose, save_interval, save_set, num_samples)

function (S::GibbsSampler)(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint};
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    verbose, save_interval, num_samples = S.verbose, S.save_interval, S.num_samples

    if initial_assignments === :background
        initial_assignments = fill(-1, length(data))
    end

    # Initialize spike assignments.
    assignments = deepcopy(initial_assignments)
    recompute_statistics!(model, data, assignments)

    # Initialize the globals using a custom function and reset model probabilities
    gibbs_initialize_globals!(model, data, assignments)
    _reset_model_probs!(model)

    results = initialize_results(model, assignments, S)
    spike_order = collect(1:length(data))

    for s in 1:num_samples

        shuffle!(spike_order)  # Update spike assignments in random order.
        for i in spike_order

            if assignments[i] != -1
                remove_datapoint!(model, data[i], assignments[i])
            else
                remove_bkgd_datapoint!(model, data[i])
            end

            assignments[i] = gibbs_add_datapoint!(model, data[i])
        end

        # Update latent events and global variables
        gibbs_sample_latents!(model)
        gibbs_sample_globals!(model, data, assignments)

        _reset_model_probs!(model)  # Recompute background and new cluster probabilities
        recompute_statistics!(model, data, assignments)  # Recompute event statistics

        # Store results
        if (s % save_interval) == 0
            j = Int(s / save_interval)
            update_results!(results, model, assignments, data, S)
            verbose && print(s, "-")  # Display progress
        end
    end
    verbose && println("Done") # Finish progress bar.

    return results
end

"""
Adds spikes `s` to an existing sequence event, to a new sequence event,
or to the background process.
For each sequence event k = 1 ... K, we compute
    prob[k] = p(x_i | z_i = k, x_{neq i}) * (N_k + alpha)
The probability of forming a new cluster is
    prob[K + 1] propto p(x_i | z_i = K + 1) * alpha * (V(K + 1) / V(K))
where p(x_i | z_i = K + 1) is the marginal probability of a singleton cluster.
See section 6 of Miller & Harrison (2018).
The probability of the background is
    prob[K + 2] propto p(x_i | bkgd) * lambda0 * (1 + beta)
where lambda0 = m.bkgd_rate and (alpha, beta) are the shape and 
rate parameters of the gamma prior on sequence event amplitudes.
"""
function gibbs_add_datapoint!(model::NeymanScottModel, x::AbstractDatapoint)

    # Create log-probability vector to sample assignments.
    #
    #  - We need to sample K + 2 possibilities. There are K existing clusters
    #    we could assign to. We could also form a new cluster (index K + 1),
    #    or assign the spike to the background (index K + 2).

    # Shape and rate parameters of gamma prior on latent event amplitude.
    α = event_amplitude(model.priors).α
    β = event_amplitude(model.priors).β

    K = num_events(model)
    
    # Grab vector without allocating new memory.
    log_probs = resize!(model.K_buffer, K + 2)

    # Iterate over model events, indexed by k = {1, 2, ..., K}.
    for (k, event) in enumerate(events(model))

        # Check if event is too far away to be considered. If sampled_type < 0,
        # then the event timestamp hasn't been sampled yet, so we can't give
        # up yet. Be aware that this previously led to a subtle bug, which was
        # very painful to fix.
        if too_far(x, event, model) && been_sampled(event)
            @debug "Too far!"
            log_probs[k] = -Inf

        # Compute probability of adding spike to cluster k.
        else
            Nk = datapoint_count(event)
            log_probs[k] = log(Nk + α) + log_posterior_predictive(event, x, model)
        end
    end
    
    # New cluster probability.
    log_probs[K + 1] = model.new_cluster_log_prob + log_posterior_predictive(x, model)

    # Background probability
    log_probs[K + 2] = model.bkgd_log_prob + bkgd_log_like(model, x)

    # Sample new assignment for spike x.
    z = sample_logprobs!(log_probs)

    # New sample corresponds to background, do nothing.
    if z == (K + 2)
        add_bkgd_datapoint!(model, x)
        return -1

    # New sample corresponds to new sequence event / cluster.
    elseif z == (K + 1)
        return add_event!(model, x)  # returns new assignment

    # Otherwise, add datapoint to existing sequence event. Note
    # that z is an integer in [1:K], while assignment indices
    # can be larger and non-contiguous.
    else
        k = events(model).indices[z]  # look up assignment index.
        add_datapoint!(model, x, k)
        return k
    end
end


"""
Resamples latent event variables.
"""
function gibbs_sample_latents!(model::NeymanScottModel)
    for event in events(model)
        gibbs_sample_event!(event, model)
    end
end