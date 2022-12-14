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

struct RealObservation{N} <: AbstractDatapoint{N}
    position::SVector{N, Float64}
end

mutable struct GaussianCluster{N} <: AbstractCluster{N}
    datapoint_count::Int
    first_moment::SVector{N, Float64}
    second_moment::SMatrix{N, N, Float64}

    _predictive_normalizer::Float64
    _predictive_mean::SVector{N, Float64}
    _predictive_variance::SMatrix{N, N, Float64}

    sampled_amplitude
    sampled_position::SVector{N, Float64}
    sampled_covariance::SMatrix{N, N, Float64}
end

mutable struct GaussianGlobals <: AbstractGlobals
    bkgd_rate::Float64
end

mutable struct GaussianPriors <: AbstractPriors

    # TODO -- these three params are common to all observation models.
    cluster_rate::Float64    
    cluster_amplitude::RateGamma
    bkgd_amplitude::RateGamma

    # TODO -- Only the params below are specific to the Gaussian case.
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

struct CircleMask{N} <: AbstractMask
    center::SVector{N, Float64}
    radius::Float64
end

struct CircleComplementMask{N} <: AbstractMask
    masks::Vector{CircleMask{N}}
    bounds::SVector{N, Float64}
end

# ===
# CONSTRUCTORS
# ===

constructor_args(e::GaussianCluster) = ()

function GaussianCluster(μ, Σ, A)
    N = length(μ)
    return GaussianCluster{N}(
        0, zeros(SVector{N}), zeros(SMatrix{N, N}), 
        0.0, zeros(SVector{N}), zeros(SMatrix{N, N}),
        A, μ, Σ,
    )
end

function GaussianCluster{N}() where {N}
    return GaussianCluster(
        0,
        zeros(SVector{N}), 
        zeros(SMatrix{N, N}),
        0.0,
        zeros(SVector{N}),
        zeros(SMatrix{N, N}),
        NOT_SAMPLED_AMPLITUDE,
        zeros(SVector{N}),
        zeros(SMatrix{N, N}),
    )
end

function GaussianPriors(
    cluster_rate,
    cluster_amplitude,
    bkgd_amplitude,
    covariance_scale,
    covariance_df,
)
    N = size(covariance_scale, 1)
    return GaussianPriors(
        cluster_rate,
        cluster_amplitude,
        bkgd_amplitude,
        covariance_scale,
        covariance_df,
        zeros(N),
        0.0,
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
    clusters = ClusterList(GaussianCluster{N}())

    # Package it all together into the model object.
    model = GaussianNeymanScottModel{N}(
        bounds,
        max_radius,
        priors,
        globals,
        clusters,
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

covariance(e::GaussianCluster) = e.sampled_covariance

function reset!(e::GaussianCluster)
    # TODO -- perhaps make a better name for this function.
    e.datapoint_count = 0
    e.first_moment = zero(e.first_moment)
    e.second_moment = zero(e.second_moment)
end

function remove_datapoint!(
    model::GaussianNeymanScottModel, 
    x::RealObservation, 
    k::Int;
    recompute_posterior::Bool=true,
    kill_empty::Bool=true,
) 
    e = clusters(model)[k]

    # If this is the last datapoint, we can return early.
    if (e.datapoint_count == 1)
        if kill_empty
            return remove_cluster!(clusters(model), k)
        else
            reset!(e)
            return k
        end
    end

    # Otherwise update the sufficient statistics.
    e.datapoint_count -= 1
    e.first_moment -= position(x)
    e.second_moment -= position(x) * position(x)'
    
    @assert e.datapoint_count >= 0

    # Recompute posterior based on new sufficient statistics.
    recompute_posterior && set_posterior!(model, k)

    return k
end

function add_datapoint!(
    model::GaussianNeymanScottModel, 
    x::RealObservation, 
    k::Int;
    recompute_posterior::Bool=true
) 
    e = clusters(model)[k]

    e.datapoint_count += 1
    e.first_moment += position(x)
    e.second_moment += position(x) * position(x)'

    recompute_posterior && set_posterior!(model, k)

    return k
end

function set_posterior!(model::GaussianNeymanScottModel, k::Int)
    e = clusters(model)[k]

    # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # Extract first and second moments
    n = e.datapoint_count
    fm = e.first_moment
    sm = e.second_moment
    dim = length(fm)
    
    # Compute number of observations
    κ0 = model.priors.mean_pseudo_obs
    ν0 = model.priors.covariance_df

    κn = n + κ0
    νn = n + ν0
    df = νn - dim + 1

    # Compute mean parameter
    μ0 = model.priors.mean_prior
    μ = (fm + κ0*μ0) / κn

    # Compute covariance parameter
    S = sm - n*μ*μ'  # Centered covariance
    Ψ0 = model.priors.covariance_scale
    Ψn = Ψ0 + S + ((κ0 * n) / κn) * (fm/n - μ0) * (fm/n - μ0)'
    Σ = (κn + 1) / (κn * (νn - dim + 1)) * Ψn

    e._predictive_mean = μ
    e._predictive_variance = Σ
    e._predictive_normalizer = _logstudent_t_normalizer(μ, Σ, df, dim)
end

function too_far(
    x::RealObservation{N}, 
    cluster::GaussianCluster{N}, 
    model::GaussianNeymanScottModel{N}
) where {N}
    distance = norm(position(cluster) - position(x))
    return distance > max_cluster_radius(model)
end




# ===
# PROBABILITIES
# ===

log_bkgd_intensity(m::GaussianNeymanScottModel, x::RealObservation) =
    log(bkgd_rate(globals(m)))

log_cluster_intensity(m::GaussianNeymanScottModel, c::GaussianCluster, x::RealObservation) =
    _logmultinormpdf(position(c), covariance(c), position(x)) + log(amplitude(c))

log_prior(model::GaussianNeymanScottModel) =
    logpdf(bkgd_amplitude(priors(model)), bkgd_rate(globals(model)))

bkgd_log_like(m::GaussianNeymanScottModel, d::RealObservation) = -log(volume(m))

log_posterior_predictive(d::RealObservation, m::GaussianNeymanScottModel) = -log(volume(m))

function log_p_latents(model::GaussianNeymanScottModel)
    priors = get_priors(model)
    globals = get_globals(model)

    # Log prior on position
    lp = -log(volume(model)) * length(clusters(model))

    for cluster in clusters(model)

        # Log prior on cluster amplitude
        lp += logpdf(cluster_amplitude(priors), cluster.sampled_amplitude)

        # Log prior on covariance
        lp += logpdf(
            InverseWishart(priors.covariance_df, priors.covariance_scale),
            cluster.sampled_covariance
        )
    end

    return lp
end

function log_posterior_predictive(
    cluster::GaussianCluster{N}, 
    x::RealObservation, 
    model::GaussianNeymanScottModel{N}
) where {N}
    # Compute number of observations
    ν0 = model.priors.covariance_df
    n = cluster.datapoint_count
    df = n + ν0 - N + 1

    # Extract cached params
    μ = cluster._predictive_mean
    Σ = cluster._predictive_variance
    Z = cluster._predictive_normalizer

    return Z + _logstudent_t_unnormalized_pdf(μ, Σ, df, N, position(x))
end

function log_marginal_event(zi, S, model::GaussianNeymanScottModel, data, assignments)
    # Fancy way --- works exactly!
    # Remove one datapoint
    i = first(S)
    x = data[i]
    remove_datapoint!(model, x, zi)
    
    # Compute p({x1}) p({x1, ..., xk} | {x1})
    ll = gauss_log_marginal(zi, model, x) + log_posterior_predictive(x, model)
    
    # Add datapoint back
    add_datapoint!(model, x, zi)

    return ll
end

function gauss_log_marginal(zi, model, x=nothing, tol=1e-2)
    C = clusters(model)[zi]

    if C.datapoint_count == 0
        return 0.0
    end

    d = length(C.first_moment)
    n = C.datapoint_count

    # Get prior mean parameters
    μ0 = model.priors.mean_prior
    κ0 = 0.0
    if isnothing(x) 
        μ0 .+= (bounds(model) ./ 2.0)
        κ0 += tol * maximum(model.bounds)
    end
    Ψ0 = model.priors.covariance_scale
    ν0 = model.priors.covariance_df

    # Get prior natural parameters
    η01, η02, η03, η04 = niw_get_natural(μ0, κ0, Ψ0, ν0, d)

    # If x != nothing, then compute p({x1, x2, ..., xk} | {x1})
    if !isnothing(x)
        # Update prior natural parameters
        η01 += 1
        η02 += x.position * x.position'
        η03 += x.position
        η04 += 1

        # Update prior mean parameters
        μ0, κ0, Ψ0, ν0 = niw_get_mean(η01, η02, η03, η04, d)
    end

    # Get posterior natural parameters
    ηN1 = η01 + n
    ηN2 = η02 + C.second_moment
    ηN3 = η03 + C.first_moment
    ηN4 = η04 + n

    # Get posterior mean parameters
    μN, κN, ΨN, νN = niw_get_mean(ηN1, ηN2, ηN3, ηN4, d)

    log_measure = (-n * d / 2) * log(2π)
    niw_log_normalizer_prior = lognorm_niw(μ0, κ0, Ψ0, ν0, d)
    niw_log_normalizer_posterior = lognorm_niw(μN, κN, ΨN, νN, d)

    return log_measure + niw_log_normalizer_posterior - niw_log_normalizer_prior
end

# ===
# SAMPLING
# ===

function sample(priors::GaussianPriors)
    # Draw background rate parameter.
    bkgd_rate = rand(priors.bkgd_amplitude)

    return GaussianGlobals(bkgd_rate)
end

"""Sample a single latent cluster from the global variables."""
function sample_cluster(::GaussianGlobals, model::GaussianNeymanScottModel{N}) where {N}
    priors = get_priors(model)
    A = rand(cluster_amplitude(priors))
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

function gibbs_sample_cluster_params!(
    cluster::GaussianCluster,
    model::GaussianNeymanScottModel{N}
) where {N}

    priors = get_priors(model)
    A0 = cluster_amplitude(priors)
    ν = priors.covariance_df
    Ψ = priors.covariance_scale

    n = datapoint_count(cluster)
    fm = cluster.first_moment
    sm = cluster.second_moment

    if n > 0
        x̄ = fm / n  # Emprical mean
        S = sm - (n * x̄ * x̄')  # Centered second moment
        Λ = ((Ψ + S) + (Ψ + S)') / 2  # Inverse Wishart posterior

        Σ = rand(InverseWishart(ν + n, Matrix(Λ)))

        cluster.sampled_amplitude = rand(posterior(n, A0))
        cluster.sampled_position = SVector{N}(rand(MultivariateNormal(x̄, Σ / n)))
        cluster.sampled_covariance = SMatrix{N, N}(Σ)

    else  # n == 0

        cluster.sampled_amplitude = rand(posterior(n, A0))
        cluster.sampled_position = SVector{N}(rand(N) .* bounds(model))
        cluster.sampled_covariance = SMatrix{N, N}(rand(InverseWishart(priors.covariance_df, priors.covariance_scale)))
        
    end

    if cluster.sampled_amplitude <= 0.0
        error()
    end
end

function gibbs_sample_globals!(
    model::GaussianNeymanScottModel, 
    data::Vector{<: RealObservation}, 
    assignments::Vector{Int}
)
    priors = get_priors(model)
    globals = get_globals(model)

    # Update background rate
    n0 = count(==(-1), assignments)
    A0 = bkgd_amplitude(priors)
    
    globals.bkgd_rate = rand(posterior(volume(model), n0, A0))
end




# ===
# MASKING
# ===

Base.in(x::RealObservation, mask::CircleMask) = 
    (norm(x.position .- mask.center) < mask.radius)

volume(mask::CircleMask{N}) where {N} = π^(N/2) * mask.radius^N / gamma(N/2 + 1)

function complement_masks(
    masks::Vector{CircleMask{N}}, 
    model::GaussianNeymanScottModel
) where {N}
    return CircleComplementMask{N}[CircleComplementMask{N}(masks, model.bounds)]
end

function create_random_mask(model::GaussianNeymanScottModel, radii::Real, pc_masked::Real)
    bounds = model.bounds
    N = length(bounds)
    volume = prod(bounds .- radii)

    @assert minimum(bounds) > radii

    # Fill box with disjoint masks
    masks = CircleMask{N}[]
    points = Iterators.product([radii:(2*radii):(M-radii) for M in bounds]...)

    for p in points
        push!(masks, CircleMask{N}(SVector(p), radii))
    end

    # Sample masks
    num_masks = floor(Int, volume * pc_masked / (π*radii^2))
    return sample(masks, num_masks, replace=false)
end

Base.in(x::RealObservation, comp_mask::CircleComplementMask) = !(x in comp_mask.masks)

function volume(complement_mask::CircleComplementMask{N}; num_samples=1000) where {N}
    bounds = complement_mask.bounds
    
    num_in_complement_mask = count(
        i -> (RealObservation{N}(rand(N) .* bounds) in complement_mask),
        1:num_samples
    )

    return prod(bounds) * (num_in_complement_mask / num_samples)
end

# ====
# HELPERS
# ====

function lognorm_niw(μ, κ, Ψ, ν, d)
    lp = 0.0

    lp += (ν * d / 2) * log(2)
    lp += Distributions.logmvgamma(d, ν/2)
    lp -= (ν / 2) * logdet(Ψ)
    lp -= (d / 2) * log(κ)
    lp += (d / 2) * log(2π)

    return lp
end

function niw_get_mean(η1, η2, η3, η4, d)
    κ = η4
    ν = η1 - d - 2
    μ = η3 / η4
    Ψ = η2 - η3*η3' / η4

    return μ, κ, Ψ, ν
end

function niw_get_natural(μ, κ, Ψ, ν, d)
    η1 = ν + d + 2
    η2 = Ψ + κ * μ * μ'
    η3 = κ * μ
    η4 = κ

    return η1, η2, η3, η4
end

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