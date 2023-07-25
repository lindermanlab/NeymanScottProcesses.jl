"""
Cables Neyman Scott Model 

Generative Model
================

Globals: 

    λ_0 ~ Gamma(α_0, β_0)

    K ~ Poisson(λ)
    K_0 ~ Poisson(λ_0)

Events:

    For k = 1, ..., K

        Each event has a node-time-mark structure.

        A_k ∼ Gamma(α, β)

        nodeproba ~ Dirichlet(a_N)

        starttime ~ Uniform([0,T]) -> improper gaussian

        markproba ~ Dirichlet(a_M)

        sigma ~ InverseChisq


Background Datapoints:
    
    For i = 1, ..., K_0

        eventnode ~ Categorical(p_B)

        eventtime ~ Uniform([0,T])

        eventmark ~ Multinomial(bgmark[eventnode])

Event Datapoints:

    For k = 1, ..., K
        # For i = 1, ..., Poisson(A_k) 

            x_node ~ Categorical(nodeproba)

            x_time ~ Gaussian(starttime, sigma)

            x_mark ~ Multinomial(markproba)

where

    λ = event rate (`priors.event_rate`)
    (α, β) = event amplitude (`priors.event_amplitude`)
    (α_0, β_0) = background amplitude (`priors.bkgd_amplitude`)
    (L_1, L_2) = bounds (`model.bounds`)

    K = number of events
    λ_0 = background rate (`globals.bkgd_rate`)
    K_0 = number of background spikes

    a_N = Dirichlet concentration parameter of prior node probability vector (priors?)
    a_M = Dirichlet concentration parameter of prior mark probability vector (priors?)

    T = total time steps (model?)
    num_entity = total number of entities (model?)
    num_words = total number of words in the vocabulary (model?)

    p_B = node probability vector for all background events (globals?)
    bgmark = background mark probability vector for all nodes (globals?)

    x_node = node index of datapoint
    x_time = time stamp of datapoint
    x_mark = word counts of datapoint

"""

struct Cable <: AbstractDatapoint{1}
    position::Float64  # point in time
    node::Int  # which embassy the cable is format
    mark::SparseVector{Int, Int}  # word counts of document 
  
    _mark_sum::Int
    # node  an index 0...totalentities
    # time  a real value in [0,T]
    # mark  a count vector (dim=total number of words)
end

num_words(c::Cable) = c._mark_sum
time(c::Cable) = c.position
Cable(position, node, mark) = Cable(position, node, mark, sum(mark))


mutable struct CableCluster <: AbstractEvent{Point}
    datapoint_count::Int
    moments::Tuple{Float64, Float64}
    node_count::Vector{Int}
    mark_count::SparseVector{Int, Int}

    # shall we save in a sparse format?
    sampled_position::Float64
    sampled_variance::Float64

    sampled_amplitude::Float64

    sampled_nodeproba::Vector{Float64}
    sampled_markproba::Vector{Float64}

    # sufficient statistics for nodeTimeMark cluster
    # mark -> this should the total word counts for all data points in
    # the cluster. i.e. the counts of datapoints for all words. 
end

mutable struct CablesGlobals <: AbstractGlobals
    bkgd_rate::Float64

    bkgd_node_prob::Vector{Float64}
    bkgd_mark_prob::Matrix{Float64}  # words x nodes
    day_of_week::Vector{Float64}

    bkgd_node_counts::Vector{Int}
    bkgd_mark_counts::Matrix{Int}  # words x nodes
    day_of_week_counts::Vector{Int}
end

mutable struct CablesPriors <: AbstractPriors
    event_rate::Float64
    event_amplitude::RateGamma
    bkgd_amplitude::RateGamma

    node_concentration::Vector{Float64}
    mark_concentration::Float64

    bkgd_node_concentration::Vector{Float64}
    bkgd_mark_concentration::Matrix{Float64}

    variance_scale::Float64
    variance_pseudo_obs::Float64

    day_of_week::Dirichlet
end

const CablesModel = NeymanScottModel{
    1, 
    Cable, 
    CableCluster, 
    CablesGlobals, 
    CablesPriors
}


const CablesEventSummary = NamedTuple{
    (:index, :position, :variance, :amplitude, :nodeproba, :markproba),
    Tuple{Int, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}}
}
