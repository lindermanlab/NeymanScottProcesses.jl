# ===
# INTERFACE
# This file specifies the interface one must implement when defining a new model.
# ===

"""Raises error for methods that need to be implemented."""
notimplemented() = error("Not yet implemented.")




# ===
# CONSTRUCTORS
# ===

"""
Returns an empty cluster. May specify arguments if desired.
"""
AbstractCluster(args...) = notimplemented()

"""
Model constructor.
"""
NeymanScottModel() = notimplemented()


# ===
# DATA MANAGEMENT
# ===

"""
Resets the sufficient statistics and sampled values of `cluster`, as if it
were empty.
"""
reset!(cluster::AbstractCluster) = notimplemented()

"""
Removes the point `x` from cluster `k` in `clusters(model)`.
"""
# remove_datapoint!(model::NeymanScottModel, x::AbstractDatapoint, k::Int64) = notimplemented()

"""
Adds the point `x` to cluster `k` in `clusters(model)`.
"""
# add_datapoint!(model::NeymanScottModel, x::AbstractDatapoint, k::Int64) = notimplemented()

# """
# (OPTIONAL) Caches information relevant to the posterior distribution.
# """
# set_posterior!(model::NeymanScottModel, k::Int) = nothing

"""
(OPTIONAL) Returns `true` if `x` is so far away from `cluster` that, with
high certainty, `cluster` is not the parent of `x`.
"""
too_far(x, cluster::AbstractCluster, model::NeymanScottModel) =
    (norm(position(cluster) .- position(x)) > max_cluster_radius(model))



# ===
# PROBABILITIES
# ===

"""
The background intensity of `x`.
"""
# log_bkgd_intensity(model::NeymanScottModel, x::AbstractDatapoint) = notimplemented()

"""
The intensity of `x` under cluster `c`.
"""
# log_cluster_intensity(model::NeymanScottModel, c::AbstractCluster, x::AbstractDatapoint) = 
#     notimplemented()

"""
Log likelihood of the latent clusters given the the global variables.

log p({z₁, ..., zₖ} | θ)
"""
log_p_latents(m::NeymanScottModel) = notimplemented()

"""
Log likelihood of the global variables given the priors.

log p(θ | η)
"""
log_prior(model::NeymanScottModel) = notimplemented()

"""
Log likelihood of `x` conditioned on assigning `x` to the background.

log p(xᵢ | ωᵢ = bkgd)
"""
# bkgd_log_like(m::NeymanScottModel, x::AbstractDatapoint) = notimplemented()

"""
Log posterior predictive probability of `x` given `e`.

log p({x} ∪ {x₁, ...,  xₖ} | {x₁, ...,  xₖ}) 
"""
# log_posterior_predictive(e::AbstractCluster, x::AbstractDatapoint, m::NeymanScottModel) = 
#     notimplemented()

"""
Log posterior predictive probability of `x` given an empty cluster `e` = {}.

log p({x} | {}) 
"""
# log_posterior_predictive(x::AbstractDatapoint, m::NeymanScottModel) = notimplemented()
  



# ===
# SAMPLING
# ===

"""
Samples an instance of the global variables from the priors.
"""
# sample(priors::ClusterPriors) = notimplemented()

"""Sample a single latent cluster from the global variables."""
sample_cluster(globals::ClusterGlobals, model::NeymanScottModel) = notimplemented()

"""Sample a datapoint from the background process."""
sample_datapoint(globals::ClusterGlobals, model::NeymanScottModel) = notimplemented()

"""Samples a datapoint from cluster 'c'."""
sample_datapoint(c::AbstractCluster, G::ClusterGlobals, M::NeymanScottModel) = notimplemented()




# ===
# GIBBS SAMPLING
# ===

"""
Sample a latent cluster given its sufficient statistics.
"""
gibbs_sample_cluster!(e::AbstractCluster, m::NeymanScottModel) = notimplemented()

"""
Sample the global variables given the data and the current sampled latent clusters.
"""
# function gibbs_sample_globals!(
#     m::NeymanScottModel, 
#     data::Vector, 
#     assignments::Vector{Int}
# )
#     return notimplemented()
# end

"""
(OPTIONAL) Update global variable sufficient statistics after removing `x` from the 
background process.
"""
add_bkgd_datapoint!(model::NeymanScottModel, x) = nothing

"""
(OPTIONAL) Update global variable sufficient statistics after adding `x` to the 
background process.
"""
remove_bkgd_datapoint!(model::NeymanScottModel, x) = nothing

"""
(OPTIONAL) Initialize global variables and their sufficient statistics, if necessary.
"""
function gibbs_initialize_globals!(
    model::NeymanScottModel, 
    data::Vector, 
    assignments::Vector{Int}
)
    return nothing
end




# ===
# MASKING
# ===
# The following methods are required to implement masked samplers.


"""
    integrated_bkgd_intensity(model, region)

Computes the integral of the background intensity function over `region`. If
the background intensity is uniform, there is no need to override this method.
"""
integrated_bkgd_intensity(model::NeymanScottModel, region::Region) =
    model.globals.bkgd_rate * volume(region)

"""
    integrated_cluster_intensity(model, cluster, region)

(OPTIONAL) Computes the integral of the intensity function induced by a
`cluster` over `region`. If left unimplemented, this will approximate the
cluster intensity using random samples.
"""
integrated_cluster_intensity(model::NeymanScottModel, cluster::AbstractCluster,  region::Region) = 
    _integrated_cluster_intensity(model, cluster, region)

function _integrated_cluster_intensity(
    model::NeymanScottModel,
    cluster::AbstractCluster,
    region::Region;
    num_samples = 1000
)
    num_in_mask = count(i -> (sample_datapoint(cluster, model) ∈ region), 1:num_samples)
    prob_in_mask = sum(num_in_mask) / num_samples
    return prob_in_mask * amplitude(cluster)
end

"""
    baseline_log_like(data, region)

(OPTIONAL) Compute log-likelihood of a baseline model within `region`. Generally,
the baseline is a homogeneous Poisson process.
"""
baseline_log_like(data::Vector, region::Region) =
    _homogeneous_baseline_log_like(data, region)

_homogeneous_baseline_log_like(data::Vector, region::Region) = (
    length(data) / volume(region)
)


"""
    create_random_mask(model::NeymanScottModel, R::Real, pc::Real)

(OPTIONAL) Create a list of randomly generated masks with length `R`, covering `pc` percent 
of the volume of `model`.
"""
create_random_mask(model::NeymanScottModel, mask_lengths::Real, percent_masked::Real) =
    notimplemented()
