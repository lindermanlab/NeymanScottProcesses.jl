"""
Log likelihood of `x` conditioned on assigning `x` to the background.

log p(xᵢ | ωᵢ = bkgd)
"""
function bkgd_log_like(m::CablesModel, x::Cable)
    # # Load buffer
    # mark_prob = m.buffers[:mark_vec]

    # mark_prob .= view(m.globals.bkgd_mark_prob, :, x.node)
    # mark_distr = SparseMultinomial(num_words(x), mark_prob)

    # @inbounds return (
    #     -log(max_time(m))  # timestamp term
    #     + log(m.globals.bkgd_node_prob[x.node])  # node prob
    #     + logpdf(mark_distr, x.mark)  # words prob
    # )

    # Load buffer
    day_of_week_distr = m.buffers[:day_of_week_distr]
    node_distribution = m.buffers[:node_buffer]
    mark_vec = m.buffers[:mark_vec]

    # Node contribution: p(n | p_n)
    @. node_distribution.p = m.priors.bkgd_node_concentration
    @. node_distribution.p += m.globals.bkgd_node_counts
    node_distribution.p ./= sum(node_distribution.p)  # Normalize
    
    # Word count contribution: p(words | word distribution)
    @. mark_vec = m.priors.bkgd_mark_concentration[:, x.node]
    @. mark_vec += m.globals.bkgd_mark_counts[:, x.node]
    mark_distribution = DirichletMultinomial(num_words(x), mark_vec)

    # Day of week contribution
    @. day_of_week_distr.p = m.priors.day_of_week.alpha
    @. day_of_week_distr.p += m.globals.day_of_week_counts
    day_of_week_distr.p ./= sum(day_of_week_distr.p)  # Normalize

    return (
        -log(max_time(m) / 7)
        + logpdf(day_of_week_distr, get_day_of_week(x.position))
        + logpdf(node_distribution, x.node)
        + logpdf(mark_distribution, x.mark) / num_words(x)
    )
end


"""
Log posterior predictive probability of `x` given `e`.

log p({x} ∪ {x₁, ...,  xₖ} | {x₁, ...,  xₖ}) 
"""
function log_posterior_predictive(
    cluster::CableCluster, 
    x::Cable, 
    model::CablesModel
)
    # TODO do we need to normalize the parameters of the DirichletMultinomial

    priors = model.priors

    # Load buffer
    node_distribution = model.buffers[:node_buffer]

    # Node contribution
    # p(n | p_n)
    @. node_distribution.p = priors.node_concentration + cluster.node_count
    node_distribution.p ./= (datapoint_count(cluster) + sum(priors.node_concentration))
    
    prob_node = logpdf(node_distribution, x.node)

    # Mark contribution
    # p(words | word distribution)
    num_samples = num_words(x)
    mark_distribution = SparseDirichletMultinomial(
        num_samples,  # number of samples
        priors.mark_concentration,  # offset
        cluster.mark_count,
    )
    prob_mark = logpdf(mark_distribution, x.mark, model.buffers[:lgamma])

    # Time contribution
    n = datapoint_count(cluster)

    # Number of psuedo-observations
    k0 = 0  # mean psuedo observations
    v0 = priors.variance_pseudo_obs  # variance psuedo observations

    kn = n + k0
    vn = n + v0

    # Compute mean parameter
    μ0 = 0
    μn = (cluster.moments[1] + k0*μ0) / kn

    # Compute variance parameter
    S = cluster.moments[2] - n*μn^2
    Ψ0 = priors.variance_scale
    Ψn = Ψ0 + S

    prob_time = logstudent_t_pdf(
        μn,
        (kn + 1) / (kn * (vn)) * Ψn,
        vn,
        time(x),
    )

    return (prob_mark / num_words(x)) + prob_node + prob_time
end 


function logstudent_t_pdf(μ::Float64, σ, ν, x)
    # Normalize to a standard (zero mean, unit variance) t-distribution
    x_standard = (x - μ) / σ
    return logpdf(TDist(ν), x_standard)
end
    

"""
Log posterior predictive probability of `x` given an empty event `e` = {}.

log p({x} | {}) 
"""
function log_posterior_predictive(
    x::Cable, 
    model::CablesModel
)
    priors = model.priors

    # Node contribution
    # p(n | p_n)
    node_distribution = model.buffers[:node_buffer]

    @. node_distribution.p = priors.node_concentration 
    node_distribution.p ./= sum(priors.node_concentration)

    logprob_node = logpdf(node_distribution, x.node)

    # Mark contribution
    # p(words | word distribution)
    mark_distribution = SparseDirichletMultinomial(
        num_words(x),
        priors.mark_concentration,
        model.buffers[:zero_mark_buffer]
    )
    logprob_mark = logpdf(mark_distribution, x.mark, model.buffers[:lgamma])

    # Time contribution
    logprob_time = -log(bounds(model))

    return logprob_node + (logprob_mark / num_words(x)) + logprob_time
end


function bkgd_intensity(model::CablesModel, x::Cable)
    # Load buffers
    mark_vec = model.buffers[:mark_vec]

    mark_vec .= view(model.globals.bkgd_mark_prob, :, x.node)
    mark_distr = SparseMultinomial(num_words(x), mark_vec)

    p = bkgd_rate(model.globals)
    p *= model.globals.bkgd_node_prob[x.node]
    p *= pdf(mark_distr, x.mark)
    p *= 7 * model.globals.day_of_week[get_day_of_week(x.position)]
    return p
end
function log_bkgd_intensity(model::CablesModel, x::Cable)
    # Load buffers
    mark_vec = model.buffers[:mark_vec]

    mark_vec .= view(model.globals.bkgd_mark_prob, :, x.node)
    mark_distr = SparseMultinomial(num_words(x), mark_vec)

    p = log(bkgd_rate(model.globals))
    p += log(model.globals.bkgd_node_prob[x.node])
    p += logpdf(mark_distr, x.mark)
    p += log(7 * model.globals.day_of_week[get_day_of_week(x.position)])
    return p
end

function event_intensity(model::CablesModel, e::CableCluster, x::Cable)
    # Load buffers
    node_distr = model.buffers[:node_buffer]

    node_distr.p .= e.sampled_nodeproba
    time_distr = Normal(e.sampled_position, e.sampled_variance)
    mark_distr = SparseMultinomial(num_words(x), e.sampled_markproba)

    p = pdf(time_distr, time(x))
    p *= pdf(node_distr, x.node)
    p *= pdf(mark_distr, x.mark)
    # @assert p > 0.0
    return p
end
function log_event_intensity(model::CablesModel, e::CableCluster, x::Cable)
    # Load buffers
    node_distr = model.buffers[:node_buffer]

    node_distr.p .= e.sampled_nodeproba
    time_distr = Normal(e.sampled_position, e.sampled_variance)
    mark_distr = SparseMultinomial(num_words(x), e.sampled_markproba)

    p = logpdf(time_distr, time(x))
    p += logpdf(node_distr, x.node)
    p += logpdf(mark_distr, x.mark)
    return p + log(amplitude(e))
end



"""
Log likelihood of the latent events given the the global variables.

log p({z₁, ..., zₖ} | θ)
"""
function log_p_latents(model::CablesModel)

    priors = model.priors
    globals = model.globals

    lp = 0.0

    # Load buffers
    mark_distr = model.buffers[:mark_dirichlet]

    for event in events(model)

        # Log prior of event amplitude
        lp += logpdf(event_amplitude(priors), position(event))

        # Log prior of event variance
        lp += logpdf(
            RateGamma(priors.variance_scale, priors.variance_pseudo_obs),
            event.sampled_variance
        )

        # Log prior of event node probabilities
        lp += logpdf(
            Dirichlet(priors.node_concentration),
            event.sampled_nodeproba
        )

        # Log prior of event mark probabilities 
        mark_distr .= priors.mark_concentration
        lp += logpdf(
            mark_distr,
            event.sampled_markproba
        )
        
    end

    # Log prior on position
    lp -= log(volume(model)) * length(events(model))

end


"""
Log likelihood of the global variables given the priors.

log p(θ | η)
"""
function log_prior(model::CablesModel)
    priors = model.priors
    globals = model.globals

    lp = 0.0

    # Rate of background spikes.
    # λ_0 ∼ Gamma(α_0, β_0)
    lp += logpdf(bkgd_amplitude(priors), bkgd_rate(globals))

    # Background node probability
    lp += logpdf(
        Dirichlet(priors.bkgd_node_concentration),
        bkgd_node_prob
    )

    # Mark node probability
    mark_distr = model.buffers[:mark_dirichlet]
    mark_vec = model.buffers[:mark_vec]
    for node in 1:num_nodes(model)
        mark_distr.alpha .= view(priors.bkgd_mark_concentration, :, node)
        mark_vec .= view(bkgd_mark_prob, :, node)
        lp += logpdf(mark_distr, mark_vec)
    end

    # Day of week probability
    lp += logpdf(priors.day_of_week, globals.day_of_week)

    return lp
end
