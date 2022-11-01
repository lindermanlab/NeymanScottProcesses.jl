function split_merge!(
    model::NeymanScottModel, 
    data, 
    assignments;
    verbose=false
)
    window_size = model.max_cluster_radius

    n = length(data)
    K_old = num_clusters(model)

    # Select two datapoints.
    cluster_spikes = findall(assignments .!= -1)
    if isempty(cluster_spikes)
        return
    end

    i = rand(cluster_spikes)
    xi, zi = data[i], assignments[i]

    # The second datapoint should be within the sampling window
    A = []
    for j in 1:n
        xj = data[j]
        if (i != j) && (assignments[j] != -1) && (norm(position(xi) - position(xj)) < window_size)
            push!(A, j)
        end
    end

    # Quit early if there aren't any neighbors nearby
    if isempty(A)
        return
    end

    j = rand(A)
    xj, zj = data[j], assignments[j]
 
    if zi == zj
        split_move!(i, j, model, data, assignments, verbose)
    else
        merge_move!(i, j, model, data, assignments, verbose)
    end
end

function merge_move!(i, j, model, data, assignments, verbose)
    zi, zj = assignments[i], assignments[j]
    xi, xj = data[i], data[j]
    K_old = num_clusters(model)

    # Compute likelihood of old partition
    Si = findall(==(zi), assignments)
    Sj = findall(==(zj), assignments)
    n1, n2 = length(Si), length(Sj)

    log_like_old = (
        log_marginal_event(zi, Si, model, data, assignments) +
        log_marginal_event(zj, Sj, model, data, assignments)
    )

    # [A] Merge all spikes into the first cluster
    for k in Sj
        remove_datapoint!(model, data[k], zj)
        add_datapoint!(model, data[k], zi)
        assignments[k] = zi
    end

    # [B] Compute acceptance probabibility
    (; α, β) = model.priors.cluster_amplitude

    # Proposal probability
    log_q_rev = (n1 + n2 - 2) * log(0.5)
    log_q_fwd = log(1.0)

    # Prior on partition
    log_p_old = (model.new_cluster_log_prob - log(α)) + lgamma(α + n1) + lgamma(α + n2) - 2*lgamma(α)
    log_p_new = lgamma(α + n1 + n2) - lgamma(α)

    # Likelihood of new partition
    log_like_new = log_marginal_event(zi, [Si; Sj], model, data, assignments)

    log_accept_prob = (
        log_q_rev - log_q_fwd
        + log_p_new - log_p_old
        + log_like_new - log_like_old
    )

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        verbose && println("Merge accepted\n")
    else
        # Undo assignments
        zj = add_cluster!(clusters(model))    
        for k in Sj
            remove_datapoint!(model, data[k], zi)
            add_datapoint!(model, data[k], zj)
            assignments[k] = zj
        end
    end
end

function split_move!(i, j, model, data, assignments, verbose)
    zi, zj = assignments[i], assignments[j]
    xi, xj = data[i], data[j]
    K_old = num_clusters(model)

    # Compute likelihood of old partition
    S_full = findall(==(zi), assignments)
    log_like_old = log_marginal_event(zi, S_full, model, data, assignments)

    # Create a new cluster and move xj over
    zj = add_cluster!(clusters(model))
    remove_datapoint!(model, xj, zi)
    add_datapoint!(model, xj, zj)
    assignments[j] = zj

    # [A] Split up remaining spikes
    S = filter(k -> (k != i) && (k != j), S_full)

    Si, Sj = [i], [j]
    for k in S
        if rand() < 0.5
            push!(Si, k)
        else
            remove_datapoint!(model, data[k], zi)
            add_datapoint!(model, data[k], zj)
            assignments[k] = zj
            push!(Sj, k)
        end
    end

    n1, n2 = length(Si), length(Sj)

    # [B] Compute acceptance probabibility
    (; α, β) = cluster_amplitude(model.priors)

    # Proposal probability
    log_q_fwd = (n1 + n2 - 2) * log(0.5)
    log_q_rev = log(1.0)

    # Prior on partition
    log_p_new = (model.new_cluster_log_prob - log(α)) + lgamma(α + n1) + lgamma(α + n2) - 2*lgamma(α)
    log_p_old = lgamma(α + n1 + n2) - lgamma(α)

    # Likelihood of new partition
    log_like_new = (
        log_marginal_event(zi, Si, model, data, assignments) + 
        log_marginal_event(zj, Sj, model, data, assignments)
    )

    log_accept_prob = (
        log_q_rev - log_q_fwd
        + log_p_new - log_p_old
        + log_like_new - log_like_old
    )


    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing        
        verbose && println("Split accepted\n")
    else
        # Undo assignments
        for k in Sj
            remove_datapoint!(model, data[k], zj)
            add_datapoint!(model, data[k], zi)
            assignments[k] = zi
        end
    end
end

function log_marginal_event(zi, S, model, data, assignments)
    return _log_marginal_event(zi, S, model, data, assignments)
end

function _log_marginal_event(zi, S, model, data, assignments)
    i = pop!(S)
    x = data[i]

    # Base case
    if length(S) == 0
        ll = log_posterior_predictive(x, model)

    # Recursive case
    else
        # Remove spike
        remove_datapoint!(model, x, zi)

        # Compute predictive probability
        ll = log_posterior_predictive(clusters(model)[zi], x, model)

        # Recurse
        ll += _log_marginal_event(zi, S, model, data, assignments)

        # Add datapoint back to cluster
        add_datapoint!(model, x, zi)
    end

    push!(S, i)

    return ll
end

function log_marginal_event(zi, S, model::GaussianNeymanScottModel, data, assignments)

    #@assert length(S) > 0
    
    # ll = _log_marginal_event(zi, S, model, data, assignments)
    # ll = gauss_log_marginal(zi, model)

    # Fancy way --- works exactly!

    # Remove one datapoint
    i = first(S)
    x = data[i]
    remove_datapoint!(model, x, zi)
    
    # Compute p({x1}) p({x1, ..., xk} | {x1})
    ll = gauss_log_marginal(zi, model, x) + log_posterior_predictive(x, model)
    
    # Add datapoint back
    add_datapoint!(model, x, zi)

    return ll
end

function gauss_log_marginal(zi, model, x=nothing, tol=1e-2)
    C = clusters(model)[zi]

    d = length(C.first_moment)
    n = C.datapoint_count

    # Get prior mean parameters
    μ0 = model.priors.mean_prior
    κ0 = 0.0
    if isnothing(x) 
        μ0 .+= (bounds(model) ./ 2.0)
        κ0 += tol * maximum(model.bounds)
    end
    Ψ0 = model.priors.covariance_scale
    ν0 = model.priors.covariance_df

    # Get prior natural parameters
    η01, η02, η03, η04 = niw_get_natural(μ0, κ0, Ψ0, ν0, d)

    # If x != nothing, then compute p({x1, x2, ..., xk} | {x1})
    if !isnothing(x)
        # Update prior natural parameters
        η01 += 1
        η02 += x.position * x.position'
        η03 += x.position
        η04 += 1

        # Update prior mean parameters
        μ0, κ0, Ψ0, ν0 = niw_get_mean(η01, η02, η03, η04, d)
    end

    # Get posterior natural parameters
    ηN1 = η01 + n
    ηN2 = η02 + C.second_moment
    ηN3 = η03 + C.first_moment
    ηN4 = η04 + n

    # Get posterior mean parameters
    μN, κN, ΨN, νN = niw_get_mean(ηN1, ηN2, ηN3, ηN4, d)

    log_measure = (-n * d / 2) * log(2π)
    niw_log_normalizer_prior = lognorm_niw(μ0, κ0, Ψ0, ν0, d)
    niw_log_normalizer_posterior = lognorm_niw(μN, κN, ΨN, νN, d)

    return log_measure + niw_log_normalizer_posterior - niw_log_normalizer_prior
end

function lognorm_niw(μ, κ, Ψ, ν, d)
    lp = 0.0

    lp += (ν * d / 2) * log(2)
    lp += Distributions.logmvgamma(d, ν/2)
    lp -= (ν / 2) * logdet(Ψ)
    lp -= (d / 2) * log(κ)
    lp += (d / 2) * log(2π)

    return lp
end

function niw_get_mean(η1, η2, η3, η4, d)
    κ = η4
    ν = η1 - d - 2
    μ = η3 / η4
    Ψ = η2 - η3*η3' / η4

    return μ, κ, Ψ, ν
end

function niw_get_natural(μ, κ, Ψ, ν, d)
    η1 = ν + d + 2
    η2 = Ψ + κ * μ * μ'
    η3 = κ * μ
    η4 = κ

    return η1, η2, η3, η4
end