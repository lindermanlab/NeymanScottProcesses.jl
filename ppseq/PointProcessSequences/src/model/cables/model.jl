const MAX_COUNT = 10_000.0


max_time(m::CablesModel) = m.bounds
volume(m::CablesModel) = m.bounds

num_nodes(m::CablesModel) = num_nodes(m.priors)
num_nodes(priors::CablesPriors) = length(priors.bkgd_node_concentration)

num_marks(m::CablesModel) = num_marks(m.priors)
num_marks(priors::CablesPriors) = size(priors.bkgd_mark_concentration, 1)

sample_num_words() = rand(Poisson(1000))

trim(x, lower, upper) = max(min(x, upper), lower)

"""
Model constructor.
"""
function CablesModel(
    max_time::Float64,
    max_radius::Float64,
    priors::CablesPriors
)
    globals = sample(priors)
    events = EventList(
        CableCluster, 
        (num_nodes(priors), num_marks(priors))
    )

    p = rand(num_nodes(priors))
    node_buffer = Categorical(p / sum(p))
    day_of_week_distr = Categorical(globals.day_of_week)

    α = priors.mark_concentration[1]

    lgamma_buffer = lgamma.(α : 1.0 : MAX_COUNT + α)

    model = CablesModel(
        max_time,
        max_radius,

        priors,
        globals,
        events,

        0.0,
        0.0,

        Float64[],
        Dict(
            :node_buffer => node_buffer,
            :day_of_week_distr => day_of_week_distr,

            :zero_mark_buffer => spzeros(num_marks(priors)),
            :mark_vec => zeros(num_marks(priors)),
            :mark_dirichlet => Dirichlet(ones(num_marks(priors))),

            :mark_node => zeros(Int, num_marks(priors), num_nodes(priors)),
            
            :lgamma => lgamma_buffer
        )
    )
    
    # Compute the background and new cluster probabilities.
    _gibbs_reset_model_probs(model)

    return model
end

"""
Samples an instance of the data from the model.
"""
function sample(
    model::CablesModel;
    resample_latents::Bool=false,
    resample_globals::Bool=false,
)
    priors = model.priors::CablesPriors

    # === Sample global parameters ===
    globals = resample_globals ? sample(priors) : deepcopy(model.globals)

    # === Sample latent events ===
    if resample_latents

        # Sample number of events
        K = rand(Poisson(event_rate(priors) * volume(model)))

        events = CablesEventSummary[]
        for k = 1:K
            A = rand(event_amplitude(priors))
            μ = rand() * max_time(model)
            σ2 = 1 / rand(RateGamma(
                priors.variance_psuedo_obs,
                priors.variance_scale
            ))
            # Sample background node probabilities
            node_prob = rand(Dirichlet(priors.node_concentration))

            # Sample background mark probabilities
            mark_prob = rand(Dirichlet(num_marks(model), priors.mark_concentration))

            push!(events, CablesEventSummary((k, μ, σ2, A, node_prob, mark_prob)))
        end

    else
        events = event_list_summary(model)
    end

    # === Sample datapoints ===
    points = Cable[]
    assignments = Int64[] 

    # Sample background datapoints
    S_bkgd = rand(Poisson(bkgd_rate(globals) * volume(model)))
    num_weeks = floor(Int, max_time(model) / 7)

    for s = 1:S_bkgd

        day = (7 * rand(1:num_weeks)) + rand(Categorical(globals.day_of_week))
        node = rand(Categorical(globals.bkgd_node_prob))
        num_words = sample_num_words()
        mark = rand(Multinomial(num_words, globals.bkgd_mark_prob[:, node]))

        push!(points, Cable(float(time), node, mark))
        push!(assignments, -1)
    end

    # Sample event-evoked datapoints
    for e in events

        # Number of datapoints evoked by latent event
        S = rand(Poisson(e.amplitude))

        # Sample datapoints
        for s = 1:S

            _time = rand(Normal(e.position, sqrt(e.variance)))
            time = trim(_time, 0, volume(model))

            node = rand(Categorical(e.nodeproba))
            num_words = sample_num_words()
            mark = rand(Multinomial(num_words, e.markproba))

            push!(points, Cable(time, node, mark))
            push!(assignments, e.index)
        end

    end

    return points, assignments, events

end


"""
Samples an instance of the global variables from the priors.
"""
function sample(priors::CablesPriors)::CablesGlobals
    # Sample background rate
    bkgd_rate = rand(priors.bkgd_amplitude)

    # Sample background node probabilities
    bkgd_node_prob = rand(Dirichlet(priors.bkgd_node_concentration))

    # Sample background mark probabilities
    bkgd_mark_prob = zeros(size(priors.bkgd_mark_concentration))
    for node in 1:num_nodes(priors)
        bkgd_mark_prob[:, node] .= rand(Dirichlet(
            priors.bkgd_mark_concentration[:, node]
        ))
    end
    
    # Sample day of week probabilities
    day_of_week = rand(priors.day_of_week)

    return CablesGlobals(bkgd_rate, bkgd_node_prob, bkgd_mark_prob, day_of_week)
end


function CablesGlobals(bkgd_rate, bkgd_node_prob, bkgd_mark_prob, day_of_week)
    return CablesGlobals(
        bkgd_rate,
        bkgd_node_prob,
        bkgd_mark_prob,
        day_of_week,
        zeros(Int, size(bkgd_node_prob)),
        zeros(Int, size(bkgd_mark_prob)),
        zeros(Int, 7),
    )
end
