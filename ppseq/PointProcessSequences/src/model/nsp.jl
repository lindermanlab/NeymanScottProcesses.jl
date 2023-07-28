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
mutable struct NeymanScottModel{
    N,
    D <: AbstractDatapoint{N},
    E <: AbstractEvent,
    G <: AbstractGlobals,
    P <: AbstractPriors
} <: AbstractModel

    bounds  # Float or tuple
    max_event_radius

    priors::P
    globals::G
    events::EventList{E}

    new_cluster_log_prob
    bkgd_log_prob

    _K_buffer::Vector
    buffers
end


priors(model::NeymanScottModel) = model.priors
globals(model::NeymanScottModel) = model.globals


function log_bkgd_intensity(model::NeymanScottModel, x::AbstractDatapoint)
    return log(bkgd_intensity(model, x))
end
function log_event_intensity(
    model::NeymanScottModel, event::AbstractEvent, x::AbstractDatapoint
)
    return log(event_intensity(model, event, x) * amplitude(event))
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
