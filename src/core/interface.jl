# ===
# INTERFACE
# This file specifies the interface one must implement when defining a new model.
# ===

"""Raises error for methods that need to be implemented."""
notimplemented() = error("Not yet implemented.")




# ===
# CONSTRUCTORS
# ===

"""
Returns an empty event. May specify arguments if desired.
"""
AbstractEvent(args...) = notimplemented()

"""
Model constructor.
"""
NeymanScottModel() = notimplemented()

"""
Returns the arguments of used to generate an empty event similar to `event`.

This is helpful when, for example, different instances of the model require
slightly different structures (for example, in a neuroscience dataset
the number of neurons will determine the size of many arrays in the event).
"""
constructor_args(event::AbstractEvent) = notimplemented()




# ===
# DATA MANAGEMENT
# ===

"""
Resets the sufficient statistics and sampled values of `event`, as if it
were empty.
"""
reset!(event::AbstractEvent) = notimplemented()

"""
Removes the point `x` from event `k` in `events(model)`.
"""
remove_datapoint!(model::NeymanScottModel, x::AbstractDatapoint, k::Int64) = notimplemented()

"""
Adds the point `x` to event `k` in `events(model)`.
"""
add_datapoint!(model::NeymanScottModel, x::AbstractDatapoint, k::Int64) = notimplemented()

"""
(OPTIONAL) Caches information relevant to the posterior distribution.
"""
set_posterior!(model::NeymanScottModel, k::Int) = nothing

"""
(OPTIONAL) Returns `true` if `x` is so far away from `event` that, with
high certainty, `event` is not the parent of `x`.
"""
too_far(x::AbstractDatapoint, event::AbstractEvent, model::NeymanScottModel) =
    (norm(position(event) .- position(x)) > max_event_radius(model))

"""
(OPTIONAL) Summarize events and return a list of simpler structs 
or named tuples.
"""
event_list_summary(model::NeymanScottModel) = [e for e in events(model)]




# ===
# PROBABILITIES
# ===

"""
The background intensity of `x`.
"""
log_bkgd_intensity(model::NeymanScottModel, x::AbstractDatapoint) = notimplemented()

"""
The intensity of `x` under event `e`.
"""
log_event_intensity(model::NeymanScottModel, e::AbstractEvent, x::AbstractDatapoint) = 
    notimplemented()

"""
Log likelihood of the latent events given the the global variables.

log p({z₁, ..., zₖ} | θ)
"""
log_p_latents(m::NeymanScottModel) = notimplemented()

"""
Log likelihood of the global variables given the priors.

log p(θ | η)
"""
log_prior(model::NeymanScottModel) = notimplemented()

"""
Log likelihood of `x` conditioned on assigning `x` to the background.

log p(xᵢ | ωᵢ = bkgd)
"""
bkgd_log_like(m::NeymanScottModel, x::AbstractDatapoint) = notimplemented()

"""
Log posterior predictive probability of `x` given `e`.

log p({x} ∪ {x₁, ...,  xₖ} | {x₁, ...,  xₖ}) 
"""
log_posterior_predictive(e::AbstractEvent, x::AbstractDatapoint, m::NeymanScottModel) = 
    notimplemented()

"""
Log posterior predictive probability of `x` given an empty event `e` = {}.

log p({x} | {}) 
"""
log_posterior_predictive(x::AbstractDatapoint, m::NeymanScottModel) = notimplemented()
  



# ===
# SAMPLING
# ===

"""
Samples an instance of the global variables from the priors.
"""
sample(priors::AbstractPriors) = notimplemented()

"""Sample a single latent event from the global variables."""
sample_event(globals::AbstractGlobals, model::NeymanScottModel) = notimplemented()

"""Sample a datapoint from the background process."""
sample_datapoint(globals::AbstractGlobals, model::NeymanScottModel) = notimplemented()

"""Samples a datapoint from event 'e'."""
sample_datapoint(e::AbstractEvent, G::AbstractGlobals, M::NeymanScottModel) = notimplemented()




# ===
# GIBBS SAMPLING
# ===

"""
Sample a latent event given its sufficient statistics.
"""
gibbs_sample_event!(e::AbstractEvent, m::NeymanScottModel) = notimplemented()

"""
Sample the global variables given the data and the current sampled latent events.
"""
function gibbs_sample_globals!(
    m::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint}, 
    assignments::Vector{Int}
)
    return notimplemented()
end

"""
(OPTIONAL) Update global variable sufficient statistics after removing `x` from the 
background process.
"""
add_bkgd_datapoint!(model::NeymanScottModel, x::AbstractDatapoint) = nothing

"""
(OPTIONAL) Update global variable sufficient statistics after adding `x` to the 
background process.
"""
remove_bkgd_datapoint!(model::NeymanScottModel, x::AbstractDatapoint) = nothing

"""
(OPTIONAL) Initialize global variables and their sufficient statistics, if necessary.
"""
function gibbs_initialize_globals!(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint}, 
    assignments::Vector{Int}
)
    return nothing
end






