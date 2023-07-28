
add_background_datapoint!(model::AbstractModel, x::AbstractDatapoint) = nothing
initialize_globals!(model::AbstractModel, datapoints, assignments) = nothing
remove_bkgd_datapoint!(model::AbstractModel, x) = nothing

abstract type AbstractSampler end

struct GibbsSampler <: AbstractSampler
    num_samples::Int
    extra_split_merge_moves::Int64
    split_merge_window::Float64
    save_every::Int64
    verbose::Bool
    save_set
    H
end
function GibbsSampler(
    ; num_samples::Int64=100,
    extra_split_merge_moves::Int64=0,
    split_merge_window::Float64=1.0,
    save_every::Int64=1,
    verbose::Bool=false,
    save_set=[:latents, :globals, :assignments],
    H=NSP(),
)
    return GibbsSampler(
        num_samples, extra_split_merge_moves, split_merge_window, save_every, verbose, 
        save_set, H
    )
end


"""
Run gibbs sampler.
"""
function gibbs_sample!(
    model::AbstractModel,
    datapoints::Vector{<: AbstractDatapoint},
    initial_assignments::Vector{Int64};
    num_samples::Int64=100,
    extra_split_merge_moves::Int64=0,
    split_merge_window::Float64=1.0,
    save_every::Int64=1,
    verbose::Bool=false,
    save_set=[:latents, :globals, :assignments],
    H=NSP(),
)
    return gibbs_sample!(
        model,
        datapoints,
        initial_assignments,
        num_samples,
        extra_split_merge_moves,
        split_merge_window,
        save_every;
        verbose=verbose,
        save_set=save_set,
    )
end

function gibbs_sample!(
        model::AbstractModel,
        datapoints::Vector{<: AbstractDatapoint},
        initial_assignments::Vector{Int64},
        num_samples::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=false,
        save_set=[:latents, :globals, :assignments],
        H=NSP(),
    )
    sampler = GibbsSampler(
        num_samples, extra_split_merge_moves, split_merge_window, save_every, verbose, 
        save_set, H
    )

    return sample!(model, datapoints, initial_assignments, sampler)
end


function sample!(
    model::AbstractModel, 
    datapoints::Vector{<: AbstractDatapoint}, 
    initial_assignments::Vector{Int64},
    S::GibbsSampler
)
    # Initialize spike assignments.
    assignments = initial_assignments
    recompute!(model, datapoints, assignments)

    # Update globals
    # Don't do this, since this will artificially inflate things like
    # the background rate if all the assingments are to the background.
    # gibbs_update_globals!(model, datapoints, assignments)

    # Instead, initialize the globals using a custom function
    initialize_globals!(model, datapoints, assignments)

    # Make sure model probabilities are correct
    _gibbs_reset_model_probs(model)

    # Things to save
    n_saved_samples = Int(round(S.num_samples / S.save_every))    
    log_p_hist = zeros(n_saved_samples)
    assignment_hist = zeros(Int64, length(datapoints), n_saved_samples)
    latent_event_hist = Any[]
    globals_hist = Any[]

    # Order to iterate over spikes.
    spike_order = collect(1:length(datapoints))

    # ======== MAIN LOOP ======== #

    for s = 1:S.num_samples

        # Update spike assignments in random order.
        Random.shuffle!(spike_order)
        for i = spike_order
            if assignments[i] != -1
                remove_datapoint!(model, datapoints[i], assignments[i])
            else
                remove_bkgd_datapoint!(model, datapoints[i])
            end
            assignments[i] = gibbs_add_datapoint!(model, datapoints[i], S.H)
        end

        # Add extra split merge moves.
        if S.extra_split_merge_moves != 0
            split_merge_sample!(
                model,
                datapoints,
                S.extra_split_merge_moves,
                assignments,
                S.split_merge_window,
            )
        end

        # Update latent events.
        gibbs_update_latents!(model)
        
        # Update globals
        gibbs_update_globals!(model, datapoints, assignments)

        # Make sure model probabilities are correct after updating globals
        _gibbs_reset_model_probs(model)

        # Recompute sufficient statistics
        recompute!(model, datapoints, assignments)

        # Store results
        if (s % S.save_every) == 0

            # Index into hist vectors.
            j = Int(s / S.save_every)

            save_sample!(
                log_p_hist, assignment_hist, latent_event_hist, globals_hist,
                j, model, datapoints, assignments, S.save_set
            )
            
            # Display progress.
            S.verbose && print(s, "-")
        end
    end

    # Finish progress bar.
    S.verbose && (n_saved_samples > 0) && println("Done")

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )
end


function save_sample!(
    log_p_hist,
    assignment_hist,
    latent_event_hist,
    globals_hist,
    j,
    model,
    datapoints,
    assignments,
    save_set
)
    log_p_hist[j] = log_like(model, datapoints)

    if :assignments in save_set
        assignment_hist[:, j] .= assignments
    end
    if :latents in save_set
        push!(latent_event_hist, event_list_summary(model))
    end
    if :globals in save_set
        push!(globals_hist, deepcopy(model.globals))
    end
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
function gibbs_add_datapoint!(model::AbstractModel, x::AbstractDatapoint, H)

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
    log_probs = resize!(model._K_buffer, K + 2)

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
            log_probs[k] = (
                cluster_prob(H, model, event)
                + log_posterior_predictive(event, x, model)
            )
        end
    end
    
    # New cluster probability.
    log_probs[K + 1] = new_cluster_prob(H, model) + log_posterior_predictive(x, model)

    # Background probability
    log_probs[K + 2] = bkgd_prob(H, model) + bkgd_log_like(model, x)

    # Sample new assignment for spike x.
    z = sample_logprobs!(log_probs)

    # New sample corresponds to background, do nothing.
    if z == (K + 2)
        add_background_datapoint!(model, x)
        return -1

    # New sample corresponds to new sequence event / cluster.
    elseif z == (K + 1)
        return add_event!(model, x)  # returns new assignment

    # Otherwise, add datapoint to existing sequence event. Note
    # that z is an integer in [1:K], while assignment indices
    # can be larger and non-contiguous.
    else
        k = events(model).indices[z]  # look up assignment index.
        return add_datapoint!(model, x, k)
    end

end


"""
Resamples sequence type, timestamp, and amplitude. For all latent events.
"""
function gibbs_update_latents!(model::AbstractModel)
    for event in events(model)
        gibbs_sample_event!(event, model)
    end
end


"""Called at the end of Gibbs update globals."""
function _gibbs_reset_model_probs(model::NeymanScottModel)
    priors = model.priors
    globals = model.globals

    α = event_amplitude(priors).α
    β = event_amplitude(priors).β

    model.new_cluster_log_prob = (
        log(α)
        + log(event_rate(priors))
        + log(volume(model))
        + α * (log(β) - log(1 + β))
    )

    model.bkgd_log_prob = (
        log(bkgd_rate(globals))
        + log(volume(model))
        + log(1 + β)
    )
end
