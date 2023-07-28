"""
Dirichlet Process Mixture Model (DPMM).
"""
struct DPMM{P,C} <: AbstractMixtureModel{P,C}
    alpha::Float64
    cluster_params_prior::P
    assignments::Vector{Int64}
    clusters::Vector{C}
end


function cluster_log_prior(model::DPMM)

    # For each existing cluster k = 1 ... K, we compute
    #
    #   prob[k] = (N_k / (alpha + sum(N_k)))
    #
    # The probability of forming a new cluster is
    #
    #   prob[K + 1] = (alpha / (alpha + sum(N_k)))
    #
    # We ignore the constant factor of [1 / (alpha + sum(N_k)] which
    # is independent of k.

    log_probs = log.(cluster_sizes(model))
    log_probs[end] = log(model.alpha)
    return log_probs
end
