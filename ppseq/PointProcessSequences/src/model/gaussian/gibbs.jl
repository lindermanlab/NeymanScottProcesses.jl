function gibbs_update_globals!(
    model::GaussianNeymanScottModel, 
    data::Vector{Point}, 
    assignments::Vector{Int}
)
    priors = model.priors
    globals = model.globals

    # Update background rate
    num_bkgd_points = mapreduce(k->(k==-1), +, assignments)

    # Update background rate
    globals.bkgd_rate = rand(RateGamma(
            bkgd_amplitude(priors).α + num_bkgd_points,
            bkgd_amplitude(priors).β + area(model)
    ))
end


function gibbs_sample_event!(
    event::Cluster, 
    model::GaussianNeymanScottModel
)
    priors = model.priors

    @assert (datapoint_count(event) > 0)

    # Sample cluster amplitude
    event.sampled_amplitude = rand(posterior(
        datapoint_count(event),
        event_amplitude(model.priors)
    ))

    # Sample cluster mean and variance
    n = datapoint_count(event)
    ν = priors.covariance_df
    Ψ = priors.covariance_scale

    empirical_μ = event.moments[1] / n
    S = event.moments[2] - (n * empirical_μ * empirical_μ')

    Λ = ((Ψ + S) + (Ψ + S)') / 2

    Σ = sample_inverse_wishart(ν + n, Λ)
    event.sampled_position = rand(MultivariateNormal(empirical_μ, Σ / n))
    event.sampled_covariance = Σ
end


# TODO
# - Verify predictive posterior
# - Figure out the appropriate way to document everything

# WISHLIST
# - Static arrays (if performance sucks)
# - Find a way to not replicate so much sampling code
# - Refactor masked gibbs, distributed gibbs, and split merge