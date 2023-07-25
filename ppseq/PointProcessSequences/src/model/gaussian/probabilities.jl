const DIM = 2

function bkgd_log_like(m::GaussianNeymanScottModel, d::Point)
    # The background data is distributed uniformly across the bounded
    # region.
    return -log(area(m))
end


# TODO Make this fast
function log_posterior_predictive(
    event::Cluster, 
    x::Point, 
    model::GaussianNeymanScottModel
)
    # Extract first and second moments
    n = event.datapoint_count
    fm = event.moments[1]
    sm = event.moments[2]

    # Add in prior contribution
    
    # The first moment is unchanged by the uniform prior, which
    # can be interpreted as the limit of a normal distribution as the
    # precision goes to zero

    # Did this https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

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

    return logstudent_t_pdf(
        μn, 
        (κn + 1) / (κn * (νn - DIM + 1)) * Ψn, 
        νn - DIM + 1, 
        position(x)
    )
end


function log_posterior_predictive(
    d::Point, 
    m::GaussianNeymanScottModel
)
    # Since the location of the cluster is unknown, we use a uniform
    # distribution on the position of the datapoint.
    return -log(area(m))
end


function bkgd_intensity(model::GaussianNeymanScottModel, x::Point)
    return bkgd_rate(model.globals)
end


function event_intensity(
    model::GaussianNeymanScottModel, 
    event::Cluster, 
    x::Point
)
    return multinormpdf(position(event), covariance(event), position(x))
end


"""
Log probability of the latent events, given the globals and priors.
"""
function log_p_latents(model::GaussianNeymanScottModel)

    priors = model.priors
    globals = model.globals

    lp = 0.0

    for event in events(model)

        # Log prior on event amplitude
        lp += logpdf(event_amplitude(priors), position(event))

        # Log prior on covariance
        lp += logpdf(
            InverseWishart(globals.covariance_df, globals.covariance_scale),
            event.sampled_covariance
        )
    end

    # Log prior on position
    lp -= log(area(model)) * length(events(model))

    return lp
end


"""
Log probability of the global variables, given the latent variables.
"""
function log_prior(model::GaussianNeymanScottModel)

    priors = model.priors
    globals = model.globals

    lp = 0.0

    # Rate of background spikes.
    # λ_0 ∼ Gamma(α_0, β_0)
    lp += logpdf(bkgd_amplitude(priors), bkgd_rate(globals))

    return lp
end


# === Internal ===
# ================


area(m::GaussianNeymanScottModel) = (bounds(m)[1] * bounds(m)[2])


function multinormpdf(mu::Vector, sigma::Matrix, x::Vector)
    k = length(mu)
    return (
        (2π)^(-k/2) 
        * det(sigma)^(-1/2) 
        * exp((-1/2) * (x-mu)' * sigma * (x-mu))
    )
end


function logstudent_t_pdf(μ::Vector{Float64}, Σ, ν, x, d=2)
    return (
        lgamma((ν + d) / 2) 
        - lgamma(ν / 2)
        - (d/2) * log(ν)
        - (d/2) * log(π)
        - (1/2) * logdet(Σ)
        - ((ν + d) / 2) * log(1 + (1/ν) * (x - μ)' * inv(Σ) * (x - μ))
    )
end