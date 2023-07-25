"""
TODO
- Handling boundary effects
- Add covariance to log_p_latents
- Add position to PPSeq log_p_latents

Gaussian Neyman Scott Model 

Generative Model
================

Globals: 

    λ_0 ∼ Gamma(α_0, β_0)

Implicit Globals:

    K ∼ Poisson(λ)
    K_0 ∼ Poisson(λ_0)

Events:

    For k = 1, ..., K
    
        A_k ∼ Gamma(α, β)
        μ_k ∼ Uniform([0, L_1] × [0, L_2])
        Σ_k ∼ InverseWishart(ν, Ψ)

Background Datapoints:
    
    For i = 1, ..., K_0

        x_0i ∼ Uniform([0, L_1] × [0, L_2])

Event Datapoints:

    For k = 1, ..., K
        For i = 1, ..., A_k

            x_ki ∼ Normal(μ_k, Σ_k)

where

    λ = event rate (`priors.event_rate`)
    (α, β) = event amplitude (`priors.event_amplitude`)
    (α_0, β_0) = background amplitude (`priors.bkgd_amplitude`)
    (L_1, L_2) = bounds (`model.bounds`)
    Ψ = scaling of covariance matrix (`priors.covariance_scale`)
    ν = degrees of freedom of covariance matrix (`priors.covariance_df`)

    K = number of events
    λ_0 = background rate (`globals.bkgd_rate`)
    K_0 = number of background spikes

    A_k = event amplitude
    μ_k = event position
    Σ_k = event covariance

    x_ki = datapoint position

"""

struct Point <: AbstractDatapoint{2}
    position::Vector{Float64}
end

mutable struct Cluster <: AbstractEvent{Point}
    datapoint_count::Int
    moments::Tuple{Vector{Float64}, Matrix{Float64}}

    sampled_position
    sampled_covariance
    sampled_amplitude
end

covariance(e::Cluster) = e.sampled_covariance

mutable struct GaussianGlobals <: AbstractGlobals
    bkgd_rate::Float64
end

mutable struct GaussianPriors <: AbstractPriors
    event_rate::Float64
    
    event_amplitude::RateGamma
    bkgd_amplitude::RateGamma

    covariance_scale::Matrix
    covariance_df::Float64

    mean_prior::Vector
    mean_pseudo_obs::Float64
end

const GaussianNeymanScottModel = NeymanScottModel{
    2, 
    Point, 
    Cluster, 
    GaussianGlobals, 
    GaussianPriors
}

const GaussianEventSummary = NamedTuple{
    (:index, :amplitude, :position, :covariance),
    Tuple{Int, Float64, Vector, Matrix}
}