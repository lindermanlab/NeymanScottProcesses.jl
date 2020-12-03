"""
Neyman-Scott Process Model.

bounds :
    All datapoints and events occur in the N-dimensional cube 
        
        (0, bounds[1]) × ... × (0, bounds[N])

max_event_radius :
    Maximum radius of an event (used to speed up parent assignment step 
    of collapsed Gibbs sampling-- we don't compute statistics for 
    events futher away than this threshold away from the datapoint.)

priors :
    Prior distributions.

globals :
    Global variables.

events :
    List of Event structs. See `./eventlist.jl` for functionality.

_K_buffer : 
    Resized vector, holding probabilities over the number of latent events. 

buffers :
    Other buffers, which may vary with the type of Neyman-Scott model.
"""
NeymanScottModel

priors(model::NeymanScottModel) = model.priors

get_priors(model::NeymanScottModel) = priors(model)

globals(model::NeymanScottModel) = model.globals

get_globals(model::NeymanScottModel) = globals(model)

events(model::NeymanScottModel) = model.events

labels(model::NeymanScottModel) = labels(events(model))

bounds(model::NeymanScottModel) = model.bounds

max_event_radius(model::NeymanScottModel) = model.max_event_radius

num_events(model::NeymanScottModel) = length(events(model))

volume(model::NeymanScottModel) = prod(bounds(model))

first_bound(model::NeymanScottModel{N, D, E, P, G}) where {N, D, E, P, G} =
    (N > 1) ? bounds(model)[1] : bounds(model)

"""
Create a singleton latent event e = {s} containing datapoint `s` and return new 
assignment index `k`.
"""
function add_event!(model::NeymanScottModel, s::AbstractDatapoint)
    k = add_event!(events(model))  # Mark event k as non-empty.
    add_datapoint!(model, s, k)  # Add spike s to event k.
    return k
end

"""
Log likelihood of the observed data given the model `model`.

log p({x1, ..., xn} | {θ, z1, ..., zk})
"""
function log_like(model::NeymanScottModel, data::Vector{<: AbstractDatapoint})
    ll = 0.0

    for x in data
        g = log_bkgd_intensity(model, x)

        for event in events(model)
            # TODO Alex, should there be a log(amplitude(event)) here
            g = logaddexp(g, log_event_intensity(model, event, x))
        end

        ll += g
    end 
    
    ll -= bkgd_rate(model.globals) * volume(model)
    for event in events(model)
        ll -= amplitude(event)
    end
    
    return ll
end




# ===
# SAMPLING
# ===

sample_datapoint(model::NeymanScottModel) = sample_datapoint(model.globals, model)

sample_datapoint(event::AbstractEvent, model::NeymanScottModel) = 
    sample_datapoint(event, model.globals, model)

"""
Sample a set of datapoints from the background process.
"""
function sample_background(globals::AbstractGlobals, model::NeymanScottModel)
    num_samples = rand(Poisson(bkgd_rate(globals) * volume(model)))
    return [sample_datapoint(model) for _ in 1:num_samples]
end

"""
Sample a set of datapoints from an event.
"""
function sample(event::AbstractEvent, globals::AbstractGlobals, model::NeymanScottModel)     
    num_samples = rand(Poisson(event.sampled_amplitude))
    return [sample_datapoint(event, model) for _ in 1:num_samples]
end

"""
Samples an instance of the data from the model.
"""
function sample(
    model::NeymanScottModel{N, D, E, P, G}; 
    resample_latents::Bool=false, resample_globals::Bool=false,
) where {N, D, E, P, G}

    priors = get_priors(model)

    # Optionally resample globals
    globals = resample_globals ? sample(priors) : deepcopy(get_globals(model))

    # Optionally resample events
    if resample_latents 
        K = rand(Poisson(event_rate(priors) * volume(model)))
        events = E[sample_event(globals, model) for k in 1:K]
    else
        events = event_list_summary(model)
    end

    # Sample background datapoints
    spikes = sample_background(globals, model)
    assignments = [-1 for _ in 1:length(spikes)]

    # Sample event-evoked datapoints
    for (ω, e) in enumerate(events)
        event_spikes = sample(e, globals, model)
        append!(spikes, event_spikes)
        append!(assignments, [ω for _ in 1:length(event_spikes)])
    end

    return spikes, assignments, events
end





# ===
# UTILITIES
# ===

"""Reset new cluster and background probabilities."""
function _reset_model_probs!(model::NeymanScottModel)
    P = priors(model)
    G = globals(model)

    Ak = event_amplitude(P)
    α, β = Ak.α, Ak.β

    model.new_cluster_log_prob = (
        log(α)
        + log(event_rate(P))
        + log(volume(model))
        + α * (log(β) - log(1 + β))
    )

    model.bkgd_log_prob = (
        log(bkgd_rate(G))  # TODO resample this?
        + log(volume(model))
        + log(1 + β)
    )
end