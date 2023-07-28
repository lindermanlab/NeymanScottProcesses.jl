"""
Mixture of Finite Mixtures Model (MFMM).
"""
struct MFMM{P,C} <: AbstractMixtureModel{P,C}
    gamma::Float64
    k_distrib::ds.Distribution
    cluster_params_prior::P
    
    assignments::Vector{Int64}
    clusters::Vector{C}
    log_v::Vector{Float64}
end

"""
Constructs GaussianMFMM instance, precomuting the necessary
log V_n(t) coefficients.
"""
function MFMM(
        gamma::Float64,
        k_distrib::ds.Distribution,
        prior::P,
        assignments::Vector{Int64},
        clusters::Vector{C}
    ) where {P<:AbstractPrior,C<:AbstractCluster}
    
    log_v = precompute_logv(length(assignments), gamma, k_distrib)
    return MFMM(gamma, k_distrib, prior, assignments, clusters, log_v)
end


"""
Precompute the log V_n(t) coefficients as done in Miller & Harrison (2018).

V_n(t) is proportional to the probability of a partition of n datapoints
into t non-empty clusters.
"""
function precompute_logv(
    n::Integer,  # number of datapoints
    gamma::Float64,
    num_cluster_distr::ds.Distribution;
    tolerance=1e-12,
)
    log_v = fill(-Inf, n)

    for t = 1:n

        k = 1
        log_cdf = -Inf

        while (log_cdf < (-tolerance))

            # Compute log probability of k clusters.
            log_pk = ds.logpdf(num_cluster_distr, k - 1)

            # Add probability of k clusters to cumulative density.
            log_cdf = logaddexp(log_cdf, log_pk)

            # Add kth term to log_v.
            if k >= t
                log_num = lgamma(k + 1) - lgamma(k - t + 1)
                log_denom = lgamma(k * gamma + n) - lgamma(k * gamma)
                log_v[t] = logaddexp(log_v[t], log_pk + log_num - log_denom)
            end

            # Increment to k + 1 clusters.
            k = k + 1
        end
    end

    return log_v
end


function cluster_log_prior(model::MFMM)

    # For each existing cluster k = 1 ... K, we compute
    #
    #   prob[k] = (N_k + gamma)
    #
    # The probability of forming a new cluster is
    #
    #   prob[K + 1] = gamma * (V_N(K + 1) / V_N(K))
    #
    # See section 6 of Miller & Harrison (2018).

    log_probs = log.(cluster_sizes(model) .+ model.gamma)
    K = length(model.clusters) - 1  # number of occupied clusters
    log_probs[end] = log(model.gamma) + model.log_v[K + 1] - model.log_v[K]
    return log_probs
end
