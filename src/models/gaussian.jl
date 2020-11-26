"""
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




# ===
# STRUCTS
# ===

struct RealObservation{N} <: AbstractDatapoint{N}
    position::SVector{N, Float64}
end

mutable struct GaussianCluster{N} <: AbstractEvent{N}
    datapoint_count::Int
    moments::Tuple{SVector{N, Float64}, SMatrix{N, N, Float64}}

    sampled_position
    sampled_covariance
    sampled_amplitude
end

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

const GaussianNeymanScottModel{N} = NeymanScottModel{
    N, 
    RealObservation{N}, 
    GaussianCluster{N}, 
    GaussianGlobals, 
    GaussianPriors
}




# ===
# UTILITY METHODS
# ===

covariance(e::GaussianCluster) = e.sampled_covariance

function _logstudent_t_pdf(μ, Σ, ν, x, d=2)
    return (
        lgamma((ν + d) / 2) 
        - lgamma(ν / 2)
        - (d/2) * log(ν)
        - (d/2) * log(π)
        - (1/2) * logdet(Σ)
        - ((ν + d) / 2) * log(1 + (1/ν) * (x - μ)' * inv(Σ) * (x - μ))
    )
end

function _multinormpdf(μ, Σ, x)
    k = length(μ)
    return (
        (2π)^(-k/2) 
        * det(Σ)^(-1/2) 
        * exp((-1/2) * (x - μ)' * Σ * (x - μ))
    )
end




# ===
# CONSTRUCTORS
# ===

constructor_args(e::GaussianCluster) = ()

function GaussianCluster(μ, Σ, A)
    N = length(μ)
    return GaussianCluster{N}(0, (zeros(SVector{N}), zeros(SMatrix{N, N})), μ, Σ, A)
end

function GaussianCluster{N}() where {N}
    return GaussianCluster(
        0,
        (zeros(SVector{2}), zeros(SMatrix{2, 2})),
        zeros(SVector{2}),
        zeros(SMatrix{2, 2}),
        NOT_SAMPLED_AMPLITUDE
    )
end

function GaussianPriors(
    event_rate,
    event_amplitude,
    bkgd_amplitude,
    covariance_scale,
    covariance_df,
)
    N = size(covariance_scale, 1)
    return GaussianPriors(
        event_rate, event_amplitude, bkgd_amplitude, covariance_scale, covariance_df,
        zeros(N), 0.0,
    )
end

function GaussianNeymanScottModel(
    bounds,
    priors::GaussianPriors;
    max_radius::Float64=Inf
)
    N = length(bounds)
    @assert N === length(priors.mean_prior)

    globals = sample(priors)
    events = EventList(GaussianCluster{N}, ())

    # Package it all together into the model object.
    model = GaussianNeymanScottModel{N}(
        bounds,
        max_radius,
        priors,
        globals,
        events,
        0.0,
        0.0,
        Float64[],
        Dict()
    )
   
    # Compute the background and new cluster probabilities
    _reset_model_probs!(model)

    return model
end



# ===
# DATA MANAGEMENT
# ===

function reset!(e::GaussianCluster)
    e.datapoint_count = 0
    e.moments[1] .= 0
    e.moments[2] .= 0
end

function remove_datapoint!(
    model::GaussianNeymanScottModel, 
    x::RealObservation, 
    k::Int;
    recompute_posterior::Bool=true
) 
    e = events(model)[k]

    # If this is the last spike in the event, we can return early.
    (e.datapoint_count == 1) && (return remove_event!(events(model), k))

    e.datapoint_count -= 1
    e.moments[1] .-= position(x)
    e.moments[2] -= position(x) * position(x)'

    recompute_posterior && set_posterior!(model, k)

    return k
end

function add_datapoint!(
    model::GaussianNeymanScottModel, 
    x::RealObservation, 
    k::Int;
    recompute_posterior::Bool=true
) 
    e = events(model)[k]

    e.datapoint_count += 1
    e.moments[1] .+= position(x)
    e.moments[2] .+= position(x) * position(x)'

    recompute_posterior && set_posterior!(model, k)

    return k
end




# ===
# PROBABILITIES
# ===

bkgd_intensity(m::GaussianNeymanScottModel, x::RealObservation) = bkgd_rate(globals(m))

event_intensity(m::GaussianNeymanScottModel, e::GaussianCluster, x::RealObservation) =
    _multinormpdf(position(e), covariance(e), position(x))

log_prior(model::GaussianNeymanScottModel) =
    logpdf(bkgd_amplitude(priors(model)), bkgd_rate(globals(model)))

bkgd_log_like(m::GaussianNeymanScottModel, d::RealObservation) = -log(volume(m))

log_posterior_predictive(d::RealObservation, m::GaussianNeymanScottModel) = -log(area(m))

function log_p_latents(model::GaussianNeymanScottModel)
    priors = get_priors(model)
    globals = get_globals(model)

    # Log prior on position
    lp = -log(volume(model)) * length(events(model))

    for event in events(model)

        # Log prior on event amplitude
        lp += logpdf(event_amplitude(priors), position(event))

        # Log prior on covariance
        lp += logpdf(
            InverseWishart(globals.covariance_df, globals.covariance_scale),
            event.sampled_covariance
        )
    end

    return lp
end

function log_posterior_predictive(
    event::GaussianCluster, 
    x::RealObservation, 
    model::GaussianNeymanScottModel
)
    # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # Extract first and second moments
    dim = length(fm)
    n = event.datapoint_count
    fm = event.moments[1]
    sm = event.moments[2]
    
    # Compute number of observations
    κ0 = model.priors.mean_pseudo_obs
    ν0 = model.priors.covariance_df

    κn = n + κ0
    νn = n + ν0

    # Compute mean parameter
    μ0 = model.priors.mean_prior
    μn = (fm + κ0*μ0) / κn

    # Compute covariance parameter
    S = sm - n*μn*μn'  # Centered covariance
    Ψ0 = model.priors.covariance_scale
    Ψn = Ψ0 + S + ((κ0 * n) / κn) * (fm/n - μ0) * (fm/n - μ0)'

    return _logstudent_t_pdf(
        μn, 
        (κn + 1) / (κn * (νn - dim + 1)) * Ψn, 
        νn - dim + 1, 
        position(x)
    )
end

  



# ===
# SAMPLING
# ===

function sample(priors::GaussianPriors)
    # Draw background rate parameter.
    bkgd_rate = rand(priors.bkgd_amplitude)

    return GaussianGlobals(bkgd_rate)
end

"""Sample a single latent event from the global variables."""
function sample_event(::GaussianGlobals, model::GaussianNeymanScottModel{N}) where {N}
    priors = get_priors(model)
    A = rand(event_amplitude(priors))
    μ = rand(N) .* bounds(model)
    Σ = rand(InverseWishart(priors.covariance_df, priors.covariance_scale))
    return GaussianCluster(SVector{N}(μ), SMatrix{N, N}(Σ), A)
end

"""Sample a datapoint from the background process."""
function sample_datapoint(::GaussianGlobals, model::GaussianNeymanScottModel{N}) where {N}
    x = rand(N) .* bounds(model)
    return RealObservation(SVector{N}(x))
end

function sample_datapoint(e::GaussianCluster, ::GaussianGlobals, ::GaussianNeymanScottModel{N}) where {N}
    x = rand(MultivariateNormal(position(e), Matrix(covariance(e))))
    return RealObservation(SVector{N}(x))
end




# ===
# GIBBS SAMPLING
# ===

function gibbs_sample_event!(event::GaussianCluster, model::GaussianNeymanScottModel)
    priors = get_priors(model)
    A0 = event_amplitude(priors)
    ν = priors.covariance_df
    Ψ = priors.covariance_scale

    n = datapoint_count(event)

    @assert (n > 0)

    x̄ = event.moments[1] / n  # Emprical mean
    S = event.moments[2] - (n * x̄ * x̄')  # Centered second moment
    Λ = ((Ψ + S) + (Ψ + S)') / 2  # Inverse Wishart posterior

    Σ = rand(InverseWishart(ν + n, Λ))

    event.sampled_amplitude = rand(posterior(n, A0))
    event.sampled_position = rand(MultivariateNormal(x̄, Σ / n))
    event.sampled_covariance = Σ
end

function gibbs_sample_globals!(
    model::GaussianNeymanScottModel, 
    data::Vector{RealObservation}, 
    assignments::Vector{Int}
)
    priors = get_priors(model)
    globals = get_globals(model)

    # Update background rate
    n0 = count(==(-1), assignments)

    # Update background rate
    A0 = bkgd_amplitude(priors)
    globals.bkgd_rate = rand(posterior(volume(model), n0, A0))
end
