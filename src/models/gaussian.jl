"""
Gaussian Neyman Scott Model 

Generative Model
================

Globals: 

    λ_0 ∼ Gamma(α_0, β_0)

Implicit Globals:

    K ∼ Poisson(λ)
    K_0 ∼ Poisson(λ_0)

Clusters:

    For k = 1, ..., K
    
        A_k ∼ Gamma(α, β)
        μ_k ∼ Uniform([0, L_1] × [0, L_2])
        Σ_k ∼ InverseWishart(ν, Ψ)

Background Datapoints:
    
    For i = 1, ..., K_0

        x_0i ∼ Uniform([0, L_1] × [0, L_2])

Cluster Datapoints:

    For k = 1, ..., K
        For i = 1, ..., A_k

            x_ki ∼ Normal(μ_k, Σ_k)

where

    λ = cluster rate (`priors.cluster_rate`)
    (α, β) = cluster amplitude (`priors.cluster_amplitude`)
    (α_0, β_0) = background amplitude (`priors.bkgd_amplitude`)
    (L_1, L_2) = bounds (`model.bounds`)
    Ψ = scaling of covariance matrix (`priors.covariance_scale`)
    ν = degrees of freedom of covariance matrix (`priors.covariance_df`)

    K = number of clusters
    λ_0 = background rate (`globals.bkgd_rate`)
    K_0 = number of datapoints in the background partition

    A_k = cluster amplitude
    μ_k = cluster position
    Σ_k = cluster covariance

    x_ki = datapoint position
"""




# ===
# STRUCTS
# ===
mutable struct GaussianCluster <: AbstractCluster
    datapoint_count::Int
    first_moment::Vector{Float64}
    second_moment::Matrix{Float64}

    _predictive_normalizer::Float64
    _predictive_mean::Vector{Float64}
    _predictive_variance::Matrix{Float64}

    sampled_amplitude::Float64
    sampled_position::Vector{Float64}
    sampled_covariance::Matrix{Float64}
end

# Standard hyperparameters for Normal-Inverse-Wishart distribution.
mutable struct GaussianPriors <: ClusterPriors
    covariance_scale::Matrix
    covariance_df::Float64
    mean_prior::Vector
    mean_pseudo_obs::Float64
end
get_cluster_type(::GaussianPriors) = GaussianCluster

# No global variables specific to the Gaussian cluster model.
struct GaussianGlobals <: ClusterGlobals end
sample_globals(::Region, ::GaussianPriors) = GaussianGlobals()


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

function _logmultinormpdf(μ, Σ, x)
    k = length(μ)
    return (
        (-k/2) * log(2π)
        + (-1/2) * logdet(Σ)
        + (-1/2) * (x - μ)' * inv(Σ) * (x - μ)
    )
end

function _logstudent_t_normalizer(μ, Σ, ν, dim)
    return (
        lgamma((ν + dim) / 2) 
        - lgamma(ν / 2)
        - (dim/2) * log(ν)
        - (dim/2) * log(π)
        - (1/2) * logdet(Σ)
    )
end

function _logstudent_t_unnormalized_pdf(μ, Σ, ν, dim, x)
    return - ((ν + dim) / 2) * log(1 + (1/ν) * (x - μ)' * inv(Σ) * (x - μ))
end

# ===
# CONSTRUCTORS
# ===

function GaussianCluster(μ, Σ, A)
    N = length(μ)
    return GaussianCluster(
        0,              # datapoint_count
        zeros(N),       # first_moment
        zeros(N, N),    # second_moment
        0.0,            # _predictive_normalizer
        zeros(N),       # _predictive_mean
        zeros(N, N),    # _predictive_variance
        A,              # sampled_amplitude
        μ,              # sampled_position
        Σ,              # sampled_amplitude
    )
end

function GaussianCluster(domain::R) where {R <: Region}
    N = ndims(domain)
    return GaussianCluster(
        0,                      # datapoint_count
        zeros(N),               # first_moment
        zeros(N, N),            # second_moment
        0.0,                    # _predictive_normalizer
        zeros(N),               # _predictive_mean
        zeros(N, N),            # _predictive_variance
        NOT_SAMPLED_AMPLITUDE,  # sampled_amplitude
        zeros(N),               # sampled_position
        zeros(N, N),            # sampled_covariance
    )
end


# function GaussianPriors(
#     cluster_rate,
#     cluster_amplitude,
#     bkgd_amplitude,
#     covariance_scale,
#     covariance_df,
# )
#     N = size(covariance_scale, 1)
#     return GaussianPriors(
#         cluster_rate,
#         cluster_amplitude,
#         bkgd_amplitude,
#         covariance_scale,
#         covariance_df,
#         zeros(N),
#         0.0,
#     )
# end

# function GaussianNeymanScottModel(
#     bounds,
#     priors::GaussianPriors;
#     max_radius::Float64=Inf
# )
#     N = length(bounds)
#     @assert N === length(priors.mean_prior)

#     globals = sample(priors)
#     clusters = ClusterList(GaussianCluster{N}())

#     # Package it all together into the model object.
#     model = GaussianNeymanScottModel{N}(
#         bounds,
#         max_radius,
#         priors,
#         globals,
#         clusters,
#         0.0,
#         0.0,
#         Float64[],
#         Dict()
#     )

#     return model
# end



# ===
# DATA MANAGEMENT
# ===

function Base.empty!(e::GaussianCluster)
    e.datapoint_count = 0
    fill!(e.first_moment, 0)
    fill!(e.second_moment, 0)
end

function downdate!(
    cluster::GaussianCluster,
    x::Vector{Float64}
)
    cluster.first_moment .-= x
    cluster.second_moment .-= (x * x')
end

function update!(
    cluster::GaussianCluster,
    x::Vector{Float64}
) 
    cluster.first_moment .+= x
    cluster.second_moment .+= (x * x')
end

function recompute_posterior!(cluster::GaussianCluster, priors::GaussianPriors)

    # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # Extract first and second moments
    n = cluster.datapoint_count
    fm = cluster.first_moment
    sm = cluster.second_moment
    dim = length(fm)

    # Empirical mean and covariance.
    xbar = fm / n
    S = sm - (n * (xbar * xbar'))

    # Compute number of observations
    κ0 = priors.mean_pseudo_obs
    ν0 = priors.covariance_df
    @show κ0

    κn = n + κ0
    νn = n + ν0
    df = νn - dim + 1

    # Compute mean parameter
    μ0 = priors.mean_prior
    μ = (fm + κ0*μ0) / κn

    # Compute covariance parameter
    Ψ0 = priors.covariance_scale
    Ψn = Ψ0 + S + ((κ0 * n) / κn) * (xbar - μ0) * (xbar - μ0)'
    Σ = (κn + 1) / (κn * (νn - dim + 1)) * Ψn

    cluster._predictive_mean = μ
    cluster._predictive_variance = Σ
    @show eigvals(Σ)
    cluster._predictive_normalizer = _logstudent_t_normalizer(μ, Σ, df, dim)
end

# function too_far(
#     x::AbstractVector{Float64}, 
#     cluster::GaussianCluster, 
#     model::GaussianNeymanScottModel
# )
#     return norm(position(cluster) - x) > max_cluster_radius(model)
# end




# ===
# PROBABILITIES
# ===

log_bkgd_intensity(m::NeymanScottModel, x) = log(m.globals.bkgd_rate)

log_cluster_intensity(m::NeymanScottModel, c::GaussianCluster, x::Vector{Float64}) =
    _logmultinormpdf(c.sampled_position, c.sampled_covariance, x) + log(c.sampled_amplitude)

log_prior(model::NeymanScottModel{<:GaussianCluster}) =
    logpdf(model.priors.bkgd_amplitude, model.globals.bkgd_rate)

bkgd_log_like(m::NeymanScottModel{<:GaussianCluster}, x) = -log(volume(m.domain)) # should this be scaled by background rate?

log_posterior_predictive(x, m::NeymanScottModel{<:GaussianCluster}) = -log(volume(m.domain))

function log_p_latents(model::NeymanScottModel{<:GaussianCluster})
    priors = get_priors(model)
    globals = get_globals(model)

    # Log prior on position
    lp = -log(volume(model.domain)) * length(model.cluster_list)

    for (i, cluster) in model.cluster_list

        # Log prior on cluster amplitude
        lp += logpdf(cluster_amplitude(priors), cluster.sampled_position)

        # Log prior on covariance
        lp += logpdf(
            InverseWishart(globals.covariance_df, globals.covariance_scale),
            cluster.sampled_covariance
        )
    end

    return lp
end

function log_posterior_predictive(
    cluster::GaussianCluster,
    x::AbstractVector,
    model::NeymanScottModel{<:GaussianCluster}
)

    # Compute number of observations
    N = length(x)
    ν0 = model.priors.cluster_priors.covariance_df
    n = cluster.datapoint_count
    df = n + ν0 - N + 1

    # Extract cached params
    μ = cluster._predictive_mean
    Σ = cluster._predictive_variance
    Z = cluster._predictive_normalizer

    return Z + _logstudent_t_unnormalized_pdf(μ, Σ, df, N, x)
end


# ===
# SAMPLING
# ===

function sample_latents(
    domain::Region,
    priors::GaussianPriors,
    ::GaussianGlobals,
    amplitudes::Vector{Float64}
)
    # The vector of cluster amplitudes tells us how
    # many samples we need to draw.
    num_samples = length(amplitudes)

    # Sample cluster means.
    μs = [sample(domain) for _ in 1:num_samples]

    # Sample cluster covariances.
    inv_wish = InverseWishart(
        priors.covariance_df,
        priors.covariance_scale
    )
    Σs = [rand(inv_wish) for _ in 1:num_samples]

    # Return array of cluster structs.
    return [GaussianCluster(μ, Σ, A) for (μ, Σ, A) in zip(μs, Σs, amplitudes)]
end

function sample(
        cluster::GaussianCluster,
        num_samples::Integer
    )
    distrib = MultivariateNormal(
        cluster.sampled_position,
        cluster.sampled_covariance
    )
    return [rand(distrib) for _ in 1:num_samples]
end



# ===
# GIBBS SAMPLING
# ===

function gibbs_sample_cluster_params!(
    cluster::GaussianCluster,
    priors::GaussianPriors
)

    ν = priors.covariance_df
    Ψ = priors.covariance_scale

    n = cluster.datapoint_count
    fm = cluster.first_moment
    sm = cluster.second_moment

    @assert (n > 0)

    xbar = fm / n  # Empirical mean
    S = sm - (n * xbar * xbar')  # Centered second moment
    Λ = ((Ψ + S) + (Ψ + S)') / 2  # Inverse Wishart posterior

    if minimum(eigvals(Λ)) < 0
        @show sm
        @show cluster.datapoint_count
        @show eigvals(S)
        @show eigvals(Λ)
    end
    Σ = rand(InverseWishart(ν + n, Λ))

    cluster.sampled_position = rand(MultivariateNormal(xbar, Σ / n))
    cluster.sampled_covariance = Σ
end


# Sampling cluster globals is a no-op for the Gaussian model.
function gibbs_sample_globals!(
        ::GaussianGlobals,
        ::GaussianPriors,
        ::Vector,
        ::Vector{Int}
    )
    return nothing
end


# """Reset new cluster and background probabilities."""
# function _reset_model_probs!(model::NeymanScottModel)
#     P = priors(model)
#     G = globals(model)

#     Ak = cluster_amplitude(P)
#     α, β = Ak.α, Ak.β

#     model.bkgd_log_prob = (
#         log(bkgd_rate(G))
#         + log(volume(model))
#         + log(1 + β)
#     )
#     model.new_cluster_log_prob = (
#         log(α)
#         + log(cluster_rate(P))
#         + log(volume(model))
#         + α * (log(β) - log(1 + β))
#     )

    
# end

