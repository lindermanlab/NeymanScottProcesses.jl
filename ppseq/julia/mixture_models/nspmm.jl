"""
Neyman-Scott Process Mixture Model (NSPMM).
"""
struct NSPMM{P,C} <: AbstractMixtureModel{P,C}
    
    # Dataset size.
    min_datapoints::Int64
    max_datapoints::Int64
    max_clusters::Int64

    # Prior params on the number of latent events.
    #   K ~ NegBinomial(alpha_0, logistic(beta_0 / volume))
    alpha_0::Float64
    beta_0::Float64

    # Prior params on the number of observed events.
    #   N ~ NegBinomial(K * alpha, logistic(beta))
    #   π ~ Dirichlet(alpha, ... , alpha)
    alpha::Float64
    beta::Float64

    # Rectangular bounds demarcating observed region of space.
    lower_lim::Vector{Float64}
    upper_lim::Vector{Float64}

    # Prior on cluster parameters (e.g. Normal-Inverse-Wishart).
    cluster_params_prior::P

    # Cluster assignments.
    assignments::Vector{Int64}
    clusters::Vector{C}

    # Coefficients.
    log_v::Matrix{Float64}
end

function NSPMM(
        min_datapoints::Int64,
        max_datapoints::Int64,
        max_clusters::Int64,
        alpha_0::Float64,
        beta_0::Float64,
        alpha::Float64,
        beta::Float64,
        lower_lim::Vector{Float64},
        upper_lim::Vector{Float64},
        cluster_params_prior::P,
        assignments::Vector{Int64},
        clusters::Vector{C}
    ) where {P<:AbstractPrior,C<:AbstractCluster}
    
    # === PRE-COMPUTE LOG_V COEFFICIENTS === #

    # Coefficients held in (datapoints x clusters) matrix.
    log_v = fill(
        -Inf, (max_datapoints - min_datapoints + 1, max_clusters))

    # Compute the prior probability over number of latent events.
    volume = prod(upper_lim - lower_lim)
    k_prior = ds.NegativeBinomial(alpha_0, lgc(beta_0 / volume))
    log_pk_prior = ds.logpdf.(k_prior, 1:max_clusters)
    println(k_prior)
    @assert ds.logcdf(k_prior, max_clusters) > -1e-6

    log_pk = zeros(max_clusters)

    for k = 1:max_clusters
        for n = min_datapoints:max_datapoints

            # Index into log_v
            i = n - min_datapoints + 1
        
            # Compute p(k | n) \propto p(n | k) p(k)
            log_num = log_pk_prior[k] + ds.logpdf(
                ds.NegativeBinomial(alpha * k, lgc(beta)), n)

            log_denom = -Inf
            for kk = 1:max_clusters
                log_pn_given_kk = ds.logpdf(ds.NegativeBinomial(alpha * kk, lgc(beta)), n)
                log_denom = logaddexp(log_denom, log_pk_prior[kk] + log_pn_given_kk)
            end
            
            log_pk[k] = log_num - log_denom

            # Compute coefficients.
            for kk = 1:max_clusters

                # Add kk-th term to log_v.
                if kk >= k
                    log_num = lgamma(kk + 1) - lgamma(kk - k + 1)
                    log_denom = lgamma(kk * alpha + n) - lgamma(kk * alpha)
                    log_v[i, k] = logaddexp(log_v[i, k], log_pk[kk] + log_num - log_denom)
                end
            end
        end
    end

    println(logsumexp(log_pk))
    @assert abs(logsumexp(log_pk)) < 1e-6

    return NSPMM(
        min_datapoints, max_datapoints, max_clusters,
        alpha_0, beta_0, alpha, beta,
        lower_lim, upper_lim, cluster_params_prior,
        assignments, clusters, log_v)

end


function cluster_log_prior(model::NSPMM)

    # For each existing cluster k = 1 ... K, we compute
    #
    #   prob[k] = (N_k + gamma)
    #
    # The probability of forming a new cluster is
    #
    #   prob[K + 1] = gamma * (V_N(K + 1) / V_N(K))
    #
    # See section 6 of Miller & Harrison (2018).


    K = length(model.clusters) - 1     # number of occupied clusters
    N = length(model.assignments)      # number of datapoints
    i = N - model.min_datapoints + 1   # index into log_v

    @assert i <= size(model.log_v, 1)
    @assert K < model.max_clusters

    log_probs = log.(cluster_sizes(model) .+ model.alpha)
    log_probs[end] = log(model.alpha) + model.log_v[i, K + 1] - model.log_v[i, K]

    return log_probs
end

