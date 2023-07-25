"""
Code for abstract types defining a Neyman-Scott model.
"""

"""Raises error for methods that need to be implemented."""
notimplemented() = error("Not yet implemented.")


abstract type AbstractDatapoint{N} end
abstract type AbstractEvent{T} end
abstract type AbstractGlobals end
abstract type AbstractPriors end
abstract type AbstractModel end


# === Event Management ===
# ========================
# Place this code in `events.jl`
# ========================

"""
Returns an empty event. May specify arguments if desired.
"""
AbstractEvent(args...) = notimplemented()

"""
Returns the arguments of used to generate an empty event similar to `event`.

This is helpful when, for example, different instances of the model require
slightly different structures (for example, in a neuroscience dataset
the number of neurons will determine the size of many arrays in the event).
"""
constructor_args(event::AbstractEvent) = notimplemented()

"""
Resets the sufficient statistics and sampled values of `event`, as if it
were empty.
"""
reset!(event::AbstractEvent) = notimplemented()

"""
Returns `true` if the event has already been sampled.
"""
been_sampled(event::AbstractEvent) = notimplemented()

"""
(Optional) Caches information relevant to the posterior distribution.
"""
set_posterior!(model::AbstractModel, k::Int) = nothing

"""
Removes the point `x` from event `k` in `events(model)`.
"""
remove_datapoint!(model::AbstractModel, x::AbstractDatapoint, k::Int64) = 
    notimplemented()

"""
Adds the point `x` to event `k` in `events(model)`.
"""
add_datapoint!(model::AbstractModel, x::AbstractDatapoint, k::Int64) = 
    notimplemented()

"""
(Optional) Returns `true` if `x` is so far away from `event` that, with
high certainty, `event` is not the parent of `x`.
"""
too_far(x::AbstractDatapoint, event::AbstractEvent, model::AbstractModel) =
    (norm(position(event) - position(x)) > max_event_radius(model))


"""
(Optional) Summarize events and return a list of simpler structs 
or named tuples.
"""
event_list_summary(model::AbstractModel) = [e for e in events(model)]



# === Probabilities ===
# ========================
# Place this code in `probabilities.jl`
# ========================

"""
Log likelihood of `x` conditioned on assigning `x` to the background.

log p(xᵢ | ωᵢ = bkgd)
"""
bkgd_log_like(m::AbstractModel, x::AbstractDatapoint) = notimplemented()

"""
Log posterior predictive probability of `x` given `e`.

log p({x} ∪ {x₁, ...,  xₖ} | {x₁, ...,  xₖ}) 
"""
log_posterior_predictive(e::AbstractEvent, x::AbstractDatapoint, m::AbstractModel) = 
    notimplemented()

"""
Log posterior predictive probability of `x` given an empty event `e` = {}.

log p({x} | {}) 
"""
log_posterior_predictive(x::AbstractDatapoint, m::AbstractModel) = notimplemented()

"""
The background intensity of `x`.
"""
bkgd_intensity(model::AbstractModel, x::AbstractDatapoint) = notimplemented()

"""
The intensity of `x` under event `e`.
"""
event_intensity(model::AbstractModel, e::AbstractEvent, x::AbstractDatapoint) =
    notimplemented()

"""
Log likelihood of the latent events given the the global variables.

log p({z₁, ..., zₖ} | θ)
"""
log_p_latents(m::AbstractModel) = notimplemented()

"""
Log likelihood of the global variables given the priors.

log p(θ | η)
"""
log_prior(model::AbstractModel) = notimplemented()
  

# === Model ===
# ========================
# Place this code in `model.jl`
# ========================

"""
Model constructor.
"""
AbstractModel(args...) = notimplemented()

"""
Samples an instance of the data from the model.
"""
sample(
    model::AbstractModel;
    resample_latents::Bool=false,
    resample_globals::Bool=false,
) = return notimplemented()

"""
Samples an instance of the global variables from the priors.
"""
sample(priors::AbstractPriors) = notimplemented()


# === Gibbs ===
# =============
# Place this code in `gibbs.jl`
# =============

"""
Sample the global variables given the data and the current sampled latent
events.
"""
gibbs_sample_globals!(
    m::AbstractModel, 
    data::Vector{<: AbstractDatapoint}, 
    assignments::Vector{Int}
) = notimplemented()


"""
Sample a latent event given its sufficient statistics.
"""
gibbs_sample_event!(e::AbstractEvent, m::AbstractModel) = notimplemented()






"""
Abstract type for observed datapoints.
"""
AbstractDatapoint

position(point::AbstractDatapoint) = point.position

function first_coordinate(point::AbstractDatapoint)
    if typeof(position(point)) <: Tuple
        return position(point)[1]
    else
        return position(point)
    end
end


"""
Abstract type for events.

Subtypes `E` must contain the fields:

    datapoint_count
    sampled_position
    sampled_amplitude

or override the behavior of the associated get methods.
"""
AbstractEvent

datapoint_count(event::AbstractEvent) = event.datapoint_count
position(event::AbstractEvent) = event.sampled_position
amplitude(event::AbstractEvent) = event.sampled_amplitude


function first_coordinate(event::AbstractEvent)
    if typeof(position(event)) <: Tuple
        return position(event)[1]
    else
        return position(event)
    end
end


"""
Abstract type for global variables.

Subtypes `G` must contain the fields

    bkgd_rate

or override the behavior of the associated get methods.
"""
AbstractGlobals

bkgd_rate(globals::AbstractGlobals) = globals.bkgd_rate


"""
Abstract type for priors.

Subtypes `P` must contain the fields:

    event_rate
    event_amplitude
    bkgd_amplitude

or override the behavior of the associated get methods.
"""
AbstractPriors

event_rate(priors::AbstractPriors) = priors.event_rate
event_amplitude(priors::AbstractPriors) = priors.event_amplitude
bkgd_amplitude(priors::AbstractPriors) = priors.bkgd_amplitude


"""
Abstract type for model.

Subtypes `M` must contain the fields:

    events
    bounds
    max_event_radius

or override the behavior of the associated get methods.
"""
AbstractModel


events(model::AbstractModel) = model.events
num_events(model::AbstractModel) = length(events(model))
bounds(model::AbstractModel) = model.bounds
volume(model::AbstractModel) = prod(bounds(model))
max_event_radius(model::AbstractModel) = model.max_event_radius

function first_bound(model::AbstractModel)
    if typeof(bounds(model)) <: Tuple
        return bounds(model)[1]
    else
        return bounds(model)
    end
end


"""
Create a singleton latent event e = {s} containing datapoint `s`
and return assignment index `k`.
"""
function add_event!(model::AbstractModel, s::AbstractDatapoint)
    # Mark event k as non-empty.
    k = add_event!(events(model))
    # Add spike s to event k.
    return add_datapoint!(model, s, k)
end