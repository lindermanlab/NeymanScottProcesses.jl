"""
Initialize the global sufficient statistics.
"""
function initialize_globals!(
    model::CablesModel, 
    data::Vector{Cable}, 
    assignments::Vector{Int}
)
    priors = model.priors
    globals = model.globals

    globals.bkgd_node_counts .= 0
    globals.bkgd_mark_counts .= 0
    globals.day_of_week_counts .= 0
    
    @inbounds for i in 1:length(data)
        if assignments[i] == -1
            x = data[i]
            
            globals.bkgd_node_counts[x.node] += 1
            globals.bkgd_mark_counts[:, x.node] .+= x.mark
            globals.day_of_week_counts[get_day_of_week(x.position)] += 1
        end
    end
end

function get_day_of_week(i)
    return 1 + (floor(Int, i) % 7)
end

function add_background_datapoint!(model::CablesModel, x::Cable)
    model.globals.bkgd_node_counts[x.node] += 1
    model.globals.bkgd_mark_counts[:, x.node] .+= x.mark
    model.globals.day_of_week_counts[get_day_of_week(x.position)] += 1
end

function remove_bkgd_datapoint!(model::CablesModel, x::Cable)
    model.globals.bkgd_node_counts[x.node] -= 1
    model.globals.bkgd_mark_counts[:, x.node] .-= x.mark
    model.globals.day_of_week_counts[get_day_of_week(x.position)] += 1
end


"""
Sample the global variables given the data and the current sampled latent
events.
"""
function gibbs_update_globals!(
    model::CablesModel, 
    data::Vector{Cable}, 
    assignments::Vector{Int}
)
    priors = model.priors::CablesPriors
    globals = model.globals

    # Update background rate
    num_bkgd_points = mapreduce(k->(k==-1), +, assignments)

    globals.bkgd_rate = rand(RateGamma(
            bkgd_amplitude(priors).α + num_bkgd_points,
            bkgd_amplitude(priors).β + volume(model)
    ))

    # Update background node probability
    bkgd_node_counts = zeros(Int, num_nodes(model))
    @inbounds for i in 1:length(data)
        if assignments[i] == -1
            bkgd_node_counts[data[i].node] += 1
        end
    end

    node_prob_distribution = Dirichlet(
        priors.bkgd_node_concentration + bkgd_node_counts
    )
    rand!(node_prob_distribution, globals.bkgd_node_prob)


    # Update background mark probabilities
    bkgd_mark_counts = model.buffers[:mark_node]
    fill!(bkgd_mark_counts, 0)
    for i in 1:length(data)
        if assignments[i] == -1
            x = data[i]
            bkgd_mark_counts[:, x.node] .+= x.mark
        end
    end

    mark_prob_distribution = Dirichlet(ones(num_marks(model)))
    @inbounds for node in 1:num_nodes(model)
        mark_prob_distribution.alpha .= view(priors.bkgd_mark_concentration, :, node)
        mark_prob_distribution.alpha .+= view(bkgd_mark_counts, :, node)
        
        rand!(mark_prob_distribution, view(globals.bkgd_mark_prob, :, node))
    end

    # Update day of week distribution
    day_of_week_posterior = priors.day_of_week.alpha
    day_of_week_posterior += globals.day_of_week_counts
    day_of_week_posterior = Dirichlet(day_of_week_posterior)

    globals.day_of_week .= rand(day_of_week_posterior)
end


"""
Sample a latent event given its sufficient statistics.
"""
function gibbs_sample_event!(cluster::CableCluster, m::CablesModel)
    priors = m.priors

    # Load buffers
    mark_distr = m.buffers[:mark_dirichlet]

    # Load prior parameters and sufficient statistics
    n = datapoint_count(cluster)
    h = cluster.moments[1]
    J = cluster.moments[2]
    α = priors.variance_pseudo_obs
    β = priors.variance_scale
    
    # Sample amplitude from a Gamma()
    cluster.sampled_amplitude = rand(posterior(n, event_amplitude(priors)))

    # Sampled cluster variance and mean
    # h = first moment
    # J = second moment
    # x̄ = f / n = empirical mean

    # Ψ = Σ (xᵢ- μ)² = Σ xᵢ² + μ² - 2 xᵢ μ = J + μ² - 2 μ Σ xᵢ 
    #   = J + n μ² - 2 μ h = J + (h²/n) - 2h²/n
    #   = J + h²/n

    μ = h / n
    Ψ = J - h^2 / n

    σ2 = 1 / rand(RateGamma(α + n/2, β + Ψ/2))
    
    cluster.sampled_variance = σ2
    cluster.sampled_position = μ + randn() * sqrt(σ2)

#    if sqrt(σ2) > max_time(m)/2
#        @show cluster
#        @show α β h J n μ Ψ σ2
#        @show sqrt(σ2)
#        error()
#    end

    # Sample node probability distribution
    cluster.sampled_nodeproba = rand(Dirichlet(
        priors.node_concentration + cluster.node_count
    ))

    # Sample mark probability distribution
    mark_distr.alpha .= priors.mark_concentration
    for (i, xi) in zip(cluster.mark_count.nzind, cluster.mark_count.nzval)
        mark_distr.alpha[i] += xi
    end

    cluster.sampled_markproba = rand(mark_distr)
end
