"""
Diagnostics for debugging models.
"""

"""Compute the probability of introducing a background spike."""
function background_assignment_prob(bkgd_rate, β, volume)
    return exp(log(bkgd_rate) + log(volume) + log(1 + β))
end


background_assignment_prob(model::AbstractModel) = 
    background_assignment_prob(
        bkgd_rate(model.globals), 
        event_amplitude(model.priors).β,
        volume(model)
    )


"""Compute the probability of introducing a new cluster."""
function new_cluster_assignment_prob(α, β, event_rate, volume)
    return exp(
        log(α)
        + log(event_rate)
        + log(volume)
        + α * (log(β) - log(1 + β))
    )
end


new_cluster_assignment_prob(model::AbstractModel) =
    new_cluster_assignment_prob(
        event_amplitude(model.priors).α,
        event_amplitude(model.priors).β,
        event_rate(model.priors),
        volume(model)
    )


function prob_ratio_vs_event_temp(mean, var, λ, λ0, volume, temp=1:1:100)
    results = []
    for T in temp
        A = specify_gamma(mean, var * T)

        bkgd_prob = background_assignment_prob(λ0, A.β, volume)
        new_prob = new_cluster_assignment_prob(A.α, A.β, λ, volume)

        push!(results, bkgd_prob)
    end

    return results
end

function prob_ratio_vs_bkgd_temp(mean, var, λ, λ0, volume, temp=1:1:100)
    results = []
    for T in temp
        A = specify_gamma(mean, var)

        bkgd_prob = background_assignment_prob(λ0 / T, A.β, volume)
        new_prob = new_cluster_assignment_prob(A.α, A.β, λ, volume)

        push!(results, bkgd_prob)
    end

    return results
end
