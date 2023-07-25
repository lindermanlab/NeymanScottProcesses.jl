function GaussianPriors(
    event_rate,
    event_amplitude,
    bkgd_amplitude,
    covariance_scale,
    covariance_df,
)
    return GaussianPriors(
        event_rate,
        event_amplitude::RateGamma,
        bkgd_amplitude::RateGamma,
        covariance_scale::Matrix,
        covariance_df::Float64,
        zeros(2),
        0.0,
    )
end


function GaussianNeymanScottModel(
    bounds::Tuple{Float64, Float64},
    max_radius::Float64,
    event_rate::Float64,
    event_amplitude::RateGamma,
    bkgd_amplitude::RateGamma,
    covariance_scale::Matrix,
    covariance_df::Float64
)
    priors = GaussianPriors(
        event_rate::Float64,
        event_amplitude::RateGamma,
        bkgd_amplitude::RateGamma,
        covariance_scale::Matrix,
        covariance_df::Float64,
    )

    return GaussianNeymanScottModel(bounds, max_radius, priors)
end

function GaussianNeymanScottModel(
    bounds::Tuple{Float64, Float64},
    max_radius::Float64,
    priors::GaussianPriors,
)
    globals = sample(priors)
    events = EventList(Cluster, ())

    # Compute probability of introducing a new cluster.
    α = priors.event_amplitude.α
    β = priors.event_amplitude.β
    λ = priors.event_rate

    # Package it all together into the model object.
    model = GaussianNeymanScottModel(
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
    _gibbs_reset_model_probs(model)

    return model
end

function sample(
    model::GaussianNeymanScottModel;
    resample_latents::Bool=false,
    resample_globals::Bool=false,
)
    priors = model.priors

    # === Sample global parameters ===
    globals = resample_globals ? sample(priors) : deepcopy(model.globals)

    # === Sample latent events ===

    if resample_latents

        # Sample number of events
        K = rand(Poisson(event_rate(priors) * area(model)))

        events = GaussianEventSummary[]
        for k = 1:K
            A = rand(event_amplitude(priors))
            μ = rand(2) .* bounds(model)
            Σ = sample_inverse_wishart(
                priors.covariance_df,
                priors.covariance_scale
            )
            push!(events, GaussianEventSummary((k, A, μ, Σ)))
        end

    else
        events = event_list_summary(model)
    end

    # === Sample datapoints ===
    points = Point[]
    assignments = Int64[] 

    # Sample background datapoints
    S_bkgd = rand(Poisson(bkgd_rate(globals) * area(model)))

    for s = 1:S_bkgd
        x = rand(2) .* bounds(model)
        push!(points, Point(x))
        push!(assignments, -1)
    end

    # Sample event-evoked datapoints
    for e in events

        # Number of spikes evoked by latent event
        S = rand(Poisson(e.amplitude))

        # Sample spike position
        for s = 1:S

            x = rand(MultivariateNormal(
                e.position, e.covariance
            ))

            push!(points, Point(x))
            push!(assignments, e.index)
            
        end

    end

    return points, assignments, events

end

function sample(priors::GaussianPriors)
    # Draw background rate parameter.
    bkgd_rate = rand(priors.bkgd_amplitude)

    return GaussianGlobals(bkgd_rate)
end

area(bounds::Tuple{Float64, Float64}) = bounds[1] * bounds[2]

function sample_inverse_wishart(v::Float64, Ψ::Matrix{Float64})
    return rand(InverseWishart(v, Ψ))
end
