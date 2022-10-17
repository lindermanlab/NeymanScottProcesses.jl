function split_merge!(
    model::NeymanScottModel, 
    data, 
    assignments;
    verbose=true
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
    K_old = length(unique(assignments[assignments .!= -1]))

    @assert zi != zj

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
    ℙ_K = Poisson(cluster_rate(model.priors))
    (; α, β) = model.priors.cluster_amplitude
    ℙ_n = NegativeBinomial(α, α/(α+β))

    # Proposal probability
    log_q_rev = (n1 + n2 - 2) * log(0.5)
    log_q_fwd = log(1.0)

    # Prior on partition
    # log_p_old = logpdf(ℙ_K, K_old) + logpdf(ℙ_n, n1) + logpdf(ℙ_n, n2)
    # log_p_new = logpdf(ℙ_K, K_old-1) + logpdf(ℙ_n, n1+n2)
    log_p_old = (model.new_cluster_log_prob - log(α)) + lgamma(n1) + lgamma(n2)
    log_p_new = lgamma(n1+n2)

    # Likelihood of new partition
    log_like_new = log_marginal_event(zi, Si, model, data, assignments)

    log_accept_prob = (
        log_q_rev - log_q_fwd
        + log_p_new - log_p_old
        + log_like_new - log_like_old
    )

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        verbose && println("Merge accepted")
        return
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
    K_old = length(unique(assignments[assignments .!= -1]))

    old_assignments = deepcopy(assignments)

    @assert zi == zj

    # Compute likelihood of old partition
    log_like_old = log_marginal_event(zi, findall(==(zi), assignments), model, data, assignments)

    # Create a new cluster and move xj over
    zj = add_cluster!(clusters(model))
    remove_datapoint!(model, xj, zi)
    add_datapoint!(model, xj, zj)
    assignments[j] = zj

    # [A] Split up remaining spikes
    S = filter(k -> (assignments[k] == zi) && (k != i) && (k != j), 1:length(data))

    n1, n2 = 0, 0
    for k in S
        if rand() < 0.5
            n1 += 1
        else
            remove_datapoint!(model, data[k], zi)
            add_datapoint!(model, data[k], zj)
            assignments[k] = zj
            n2 += 1
        end
    end

    # [B] Compute acceptance probabibility
    ℙ_K = Poisson(cluster_rate(model.priors))
    (; α, β) = cluster_amplitude(model.priors)
    ℙ_n = NegativeBinomial(α, α/(α+β))

    # Proposal probability
    log_q_fwd = (n1 + n2) * log(0.5)
    log_q_rev = log(1.0)

    # Prior on partition
    # log_p_new = logpdf(ℙ_K, K_old+1) + logpdf(ℙ_n, n1+1) + logpdf(ℙ_n, n2+1)
    # log_p_old = logpdf(ℙ_K, K_old) + logpdf(ℙ_n, n1+n2+2)
    log_p_new = (model.new_cluster_log_prob - log(α)) + lgamma(n1+1) + lgamma(n2+1)
    log_p_old = lgamma(n1+n2+2)

    # Likelihood of new partition
    log_like_new = (
        log_marginal_event(zi, findall(==(zi), assignments), model, data, assignments) + 
        log_marginal_event(zj, findall(==(zj), assignments), model, data, assignments)
    )

    log_accept_prob = (
        log_q_rev - log_q_fwd
        + log_p_new - log_p_old
        + log_like_new - log_like_old
    )

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        verbose && println("Split accepted")
        return
    else
        # Undo assignments
        # println("Split rejected\n")
        for k in [j; S]
            if assignments[k] == zj
                remove_datapoint!(model, data[k], zj)
                add_datapoint!(model, data[k], zi)
                assignments[k] = zi
            end
        end
    end
end

function log_marginal_event(zi, S, model, data, assignments)
    i = pop!(S)
    x = data[i]

    @assert zi == assignments[i]

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
        ll += log_marginal_event(zi, S, model, data, assignments)

        # Add datapoint back to cluster
        add_datapoint!(model, x, zi)
    end

    push!(S, i)

    return ll
end 
