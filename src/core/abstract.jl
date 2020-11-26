# ===
# Code for abstract types defining a Neyman-Scott model.
# ===

abstract type AbstractDatapoint{N} end

abstract type AbstractEvent{N} end

abstract type AbstractGlobals end

abstract type AbstractPriors end

struct EventList{E <: AbstractEvent}
    constructor
    events::Vector{E}
    indices::Vector{Int64}    # Indices of occupied sequences.
end

mutable struct NeymanScottModel{
    N,
    D <: AbstractDatapoint{N},
    E <: AbstractEvent{N},
    G <: AbstractGlobals,
    P <: AbstractPriors
}
    bounds  # Float or tuple
    max_event_radius

    priors::P
    globals::G
    events::EventList{E}

    new_cluster_log_prob
    bkgd_log_prob

    K_buffer::Vector
    buffers
end


const NOT_SAMPLED_AMPLITUDE = -1.0




"""
Abstract type for observed datapoints.
"""
AbstractDatapoint

position(point::AbstractDatapoint) = point.position

function first_coordinate(point::AbstractDatapoint{N}) where {N}
    return (N > 1) ? position(point)[1] : position(point)
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

been_sampled(e::AbstractEvent) = amplitude(e) > 0

function first_coordinate(event::AbstractEvent{N}) where {N}
    return (N > 1) ? position(event)[1] : position(event)
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