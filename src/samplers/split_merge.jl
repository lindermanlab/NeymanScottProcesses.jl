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
    # log_p_old = logpdf(ℙ_K, K_old) + logpdf(ℙ_n, n1) + logpdf(ℙ_n, n2)
    # log_p_new = logpdf(ℙ_K, K_old-1) + logpdf(ℙ_n, n1+n2)
    log_p_old = (model.new_cluster_log_prob - log(α)) + lgamma(α + n1) + lgamma(α + n2)
    log_p_new = lgamma(α + n1 + n2)

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
        println("Merge accepted")
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
    log_p_new = (model.new_cluster_log_prob - log(α)) + lgamma(α + n1) + lgamma(α + n2)
    log_p_old = lgamma(α + n1 + n2)

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

    # @show log_p_new - log_p_old
    # @show log_like_new - log_like_old
    # @show log_q_rev - log_q_fwd
    # @show exp(log_accept_prob)

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        println("Split accepted")
    else
        # Undo assignments
        # println("Split rejected\n")
        for k in Sj
            remove_datapoint!(model, data[k], zj)
            add_datapoint!(model, data[k], zi)
            assignments[k] = zi
        end
    end
    # println()
end

function log_marginal_event(zi, S, model, data, assignments)
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
        ll += log_marginal_event(zi, S, model, data, assignments)

        # Add datapoint back to cluster
        add_datapoint!(model, x, zi)
    end

    push!(S, i)

    return ll
end 
