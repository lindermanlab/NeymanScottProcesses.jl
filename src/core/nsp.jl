"""
Neyman-Scott Process Model.

bounds :
    All datapoints and cluster clusters occur in the N-dimensional cube 
        
        (0, bounds[1]) × ... × (0, bounds[N])

max_event_radius :
    Maximum radius of a cluster (used to speed up parent assignment step 
    of collapsed Gibbs sampling-- we don't compute statistics for 
    clusters futher away than this threshold away from the datapoint.)

priors :
    Prior distributions.

globals :
    Global variables.

clusters :
    List of Cluster structs. See `./cluster_list.jl` for functionality.

_K_buffer : 
    Resized vector, holding probabilities over the number of clusters. 

buffers :
    Other buffers, which may vary with the type of Neyman-Scott model.
"""
NeymanScottModel

priors(model::NeymanScottModel) = model.priors

get_priors(model::NeymanScottModel) = priors(model)

globals(model::NeymanScottModel) = model.globals

get_globals(model::NeymanScottModel) = globals(model)

clusters(model::NeymanScottModel) = model.clusters

labels(model::NeymanScottModel) = labels(clusters(model))

bounds(model::NeymanScottModel) = model.bounds

max_event_radius(model::NeymanScottModel) = model.max_event_radius

num_clusters(model::NeymanScottModel) = length(clusters(model))

volume(model::NeymanScottModel) = prod(bounds(model))

first_bound(model::NeymanScottModel{N, D, E, P, G}) where {N, D, E, P, G} =
    (N > 1) ? bounds(model)[1] : bounds(model)

"""
Create a singleton cluster containing datapoint `x` and return new 
assignment index `k`.
"""
function add_cluster!(model::NeymanScottModel, x::AbstractDatapoint)
    k = add_cluster!(clusters(model))
    add_datapoint!(model, x, k)
    return k
end

"""
Log likelihood of the observed data given the model `model`.

log p({x1, ..., xn} | {θ, z1, ..., zk})
"""
function log_like(model::NeymanScottModel, data::Vector{<: AbstractDatapoint})
    ll = 0.0

    for x in data
        g = log_bkgd_intensity(model, x)

        for cluster in clusters(model)
            g = logaddexp(g, log_cluster_intensity(model, cluster, x))
        end

        ll += g
    end 
    
    ll -= bkgd_rate(model.globals) * volume(model)
    for cluster in clusters(model)
        ll -= amplitude(cluster)
    end
    
    return ll
end




# ===
# SAMPLING
# ===

sample_datapoint(model::NeymanScottModel) = sample_datapoint(model.globals, model)

sample_datapoint(cluster::AbstractCluster, model::NeymanScottModel) = 
    sample_datapoint(cluster, model.globals, model)

"""
Sample a set of datapoints from the background process.
"""
function sample_background(globals::AbstractGlobals, model::NeymanScottModel)
    num_samples = rand(Poisson(bkgd_rate(globals) * volume(model)))
    return [sample_datapoint(model) for _ in 1:num_samples]
end

"""
Sample a set of datapoints from an cluster.
"""
function sample(cluster::AbstractCluster, globals::AbstractGlobals, model::NeymanScottModel)     
    num_samples = rand(Poisson(cluster.sampled_amplitude))
    return [sample_datapoint(cluster, model) for _ in 1:num_samples]
end

"""
Samples an instance of the data from the model.
"""
function sample(
    model::NeymanScottModel{N, D, E, P, G}; 
    resample_latents::Bool=false, resample_globals::Bool=false,
) where {N, D, E, P, G}

    priors = get_priors(model)

    # Optionally resample globals
    globals = resample_globals ? sample(priors) : deepcopy(get_globals(model))

    # Sample clusters
    if resample_latents 
        K = rand(Poisson(cluster_rate(priors) * volume(model)))
        clusters = E[sample_event(globals, model) for k in 1:K]
    else
        clusters = event_list_summary(model)
    end

    # Sample background datapoints
    datapoints = sample_background(globals, model)
    assignments = [-1 for _ in 1:length(datapoints)]

    # Sample cluster-evoked datapoints
    for (ω, e) in enumerate(clusters)
        S = rand(Poisson(amplitude(e)))
        append!(datapoints, D[sample_datapoint(e, globals, model) for _ in 1:S])
        append!(assignments, Int64[ω for _ in 1:S])
    end

    return datapoints, assignments, clusters
end





# ===
# UTILITIES
# ===

"""Reset new cluster and background probabilities."""
function _reset_model_probs!(model::NeymanScottModel)
    P = priors(model)
    G = globals(model)

    Ak = cluster_amplitude(P)
    α, β = Ak.α, Ak.β

    model.new_cluster_log_prob = (
        log(α)
        + log(cluster_rate(P))
        + log(volume(model))
        + α * (log(β) - log(1 + β))
    )

    model.bkgd_log_prob = (
        log(bkgd_rate(G))  # TODO resample this?
        + log(volume(model))
        + log(1 + β)
    )
end