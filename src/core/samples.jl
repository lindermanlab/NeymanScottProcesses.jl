# =======
#
# Sampling from the full generative process.
#
# =======

"""
    datapoints, assignments, clusters = sample_full_process(model)

Samples the full Neyman-Scott Process. Returns observed events
(datapoints), parent assignments, and latent events (clusters).
"""
function sample_full_process(model::NeymanScottModel)

    # Sample global variables from prior.
    globals = sample_globals(model.domain, model.priors)

    # Sample latent events. The function `sample_latents`
    # must be implemented for each clustering model.
    clusters = sample_latents(
        model.domain,
        model.priors,
        globals
    )

    # Sample observations with assignments.
    datapoints, assignments = sample_observations(
        model.domain, model.priors, globals, clusters
    )

    # Return observed datapoints, assignments, and latent events.
    return datapoints, assignments, clusters
end


# =======
#
# Sampling global variables
#
# =======

"""
    globals = sample_globals(domain, priors)

Samples global variables.
"""
function sample_globals(
        domain::Region,
        priors::NeymanScottPriors{C}
    ) where {C <: AbstractCluster}

    bkgd_rate = rand(priors.bkgd_amplitude)
    bkgd_log_prob = (
        log(bkgd_rate)
        + log(volume(domain))
        + log(1 + priors.cluster_amplitude.β)
    )
    cluster_globals = sample_globals(domain, priors.cluster_priors)

    return NeymanScottGlobals{C}(
        bkgd_rate,
        bkgd_log_prob,
        cluster_globals
    )
end


# =======
#
# Sampling latent events, conditioned on global variables.
#
# =======

"""
    globals = sample_globals(domain, priors, globals)

Samples set of latent events (i.e. number of clusters and their
parameters).
"""
function sample_latents(
    domain::Region,
    priors::NeymanScottPriors,
    globals::NeymanScottGlobals
)

    # Sample number of clusters / latent events.
    num_samples = rand(
        Poisson(priors.cluster_rate * volume(domain))
    )

    # Sample cluster amplitudes.
    cluster_amplitudes = rand(
        priors.cluster_amplitude, num_samples
    )

    # Sample additional cluster parameters
    return sample_latents(
        domain,
        priors.cluster_priors,
        globals.cluster_globals,
        cluster_amplitudes
    )

end


# =======
#
# Sampling observations, conditioned on global variables and latent events.
#
# =======

function sample_observations(
        domain::Region,
        priors::NeymanScottPriors,
        globals::NeymanScottGlobals,
        clist::ClusterList
    )

    # Allocate arrays.
    T = observations_type(domain)
    datapoints = T[]
    assignments = Int[]

    # Call inplace function.
    return sample_observations!(
        datapoints,
        assignments,
        domain,
        priors,
        globals,
        clist
    )
end


function sample_observations!(
        datapoints::Vector,
        assignments::Vector,
        domain::Region,
        priors::NeymanScottPriors,
        globals::NeymanScottGlobals,
        cluster_list::ClusterList
    )

    empty!(datapoints)
    empty!(assignments)

    # First, sample background events.
    n_bkgd = rand(Poisson(globals.bkgd_rate * volume(domain)))
    for i = 1:n_bkgd
        push!(datapoints, sample(domain))
        push!(assignments, -1)
    end

    # Then, sample observed events from each cluster
    for (i, cluster) in enumerate(cluster_list)

        # The assignment id for cluster.
        k = cluster_list.indices[i]
        
        # Draw samples, filter out samples not in domain.
        num_samples = rand(Poisson(cluster.sampled_amplitude))
        sampled_points = filter(
            x -> (x in domain),
            sample(cluster, num_samples)
        )

        # Note that length(samples) != num_samples, in general,
        # since samples not in domain are rejected.
        append!(datapoints, sampled_points)
        append!(assignments, fill(k, length(sampled_points)))

    end

    return datapoints, assignments

end

# Convert vector of clusters to ClusterList.
function sample_observations(
        domain::Region,
        priors::NeymanScottPriors,
        globals::NeymanScottGlobals,
        clusters::Vector{C}
    ) where {C <: AbstractCluster}
    return sample_observations(
        domain,
        priors,
        globals,
        ClusterList(clusters)
    )
end

# =======
#
# Sampling observations in masked regions.
#
# =======

"""
    sample_in_mask(model, mask::Region)

Impute missing data by drawing samples from `model`. Samples that fall inside
the region `mask` are returned. Samples that are not in the masked region
are rejected.
"""
function sample_in_mask(
    model::NeymanScottModel,
    mask::Region
)
    datapoints = observations_type(model.domain)[]
    assignments = Int64[]
    return sample_in_mask!(
        datapoints,
        assignments,
        model,
        mask
    )
end

function sample_in_mask!(
    datapoints::Vector,
    assignments::Vector,
    model::NeymanScottModel,
    mask::Region
)

    # Sample observations over the full domain.
    sample_observations!(
        datapoints,
        assignments,
        model.domain,
        model.priors,
        model.globals,
        model.cluster_list
    )
    
    # Iterate backwards over datapoints.
    for i = length(datapoints):-1:1

        # Remove datapoint `i` if it falls outside the masked
        # region. Note that future indices (i.e. those less
        # than `i` are preserved).
        if !(datapoints[i] in mask)
            deleteat!(datapoints, i)
            deleteat!(assignments, i)
        end
    end

    # Sanity check
    for x in datapoints
        @assert (x in mask)
    end

    return datapoints, assignments

end
