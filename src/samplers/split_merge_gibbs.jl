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

    # Create launch state
    # - We need to partition the data into Si, Sj where i ∈ Si and j ∈ Sj
    # - We do this by first randomly assigning data to two clusters, then
    #   applying `t` rounds of restricted Gibbs sampling.
    # - Remember to store the initial state, so we can go back if needed

    Si0, Sj0, log_prob_old = get_initial_split_move_state(i, j, assignments)


    # NOTE: This mutates both `model` and `assignments`
    Si, Sj = initialize_launch_state(i, j, Si0, Sj0, assignments)

    for _ in 1:num_gibbs
        ni, nj, _ = restricted_gibbs!(i, j, S_full, model, data, assignments)
    end
    Si, Sj = filter(==(zi), assignments), filter(==(zj), assignments)
    

    # move = (i, j, Si, Sj, S_full)
    initial_state = (Si0, Sj0, log_prob_old)

    # split_move!(move, initial_state, model, data, assignments, verbose)
    #   ... ni, nj, log_p_gibbs = restricted_gibbs!(i, j, Si, Sj, model, data, assignments) 
    #   ... # ^ mutates, and also splits the data
    #   ... if failed then merge_data!(S_full, model, data, assignments)
    # merge_move!(move, initial_state, model, data, assignments, verbose)
    #   ... log_p_gibbs = psuedo_restricted_gibbs!(Si0, Sj0, i, j, Si, Sj, model, data, assignments)
    #   ... merge_data!(S_full, model, data, assignments)
    #   ... if failed then split_data!(Si0, Sj0, model, data, assignments)
 
    if zi == zj
        split_move!(i, j, model, data, assignments, verbose)
    else
        merge_move!(i, j, model, data, assignments, verbose)
    end
end

function initialize_launch_state!(i, j, Si0, Sj0 assignments)
    S_full = Iterators.flatten((Si0, Sj0))

    Si, Sj = [i], [j]
    for k in S_full
        if (k == i) || (k == j)
            continue  # Skip
        end

        if rand() <= 0.5
            push!(Si, k)
        else
            push!(Sj, k)
        end
    end

    # TODO actually change assignments and model

    return Si, Sj
end

function get_initial_split_merge_move_state(i, j, model, data, assignments)
    (; α) = model.priors.cluster_amplitude

    zi, zj = assignments[i], assignments[j]
    Si0 = filter(==(zi), assignments)

    log_p = 0.0

    if i == j  # Initially one cluster (we will do a split move)
        Sj0 = []
    else  # i != j, Intially two clusters (we will do a merge move)
        Sj0 = filter(==(zj), assignments)
    end

    n1, n2 = length(Si0), length(Sj0)

    log_p += sm_move_unnormalized_prior(Si, Sj, model)
    log_p += log_marginal_event(zi, Si0, model, data, assignments)
    log_p += (n2 == 0) ? 0.0 : log_marginal_event(zi, Sj0, model, data, assignments)

    return Si0, Sj0, log_p
end

function sm_move_unnormalized_prior(Si, Sj, model)
    (; α) = model.priors.cluster_amplitude
    mnclp = model.new_cluster_log_prob

    n1, n2 = length(Si), length(Sj)

    if n2 == 0  # One cluster
        return lgamma(α + n1 + n2) - lgamma(α)
    else  # Two clusters
        return (mnclp - log(α)) + lgamma(α + n1) + lgamma(α + n2) - 2*lgamma(α)
    end
end 

"""
    restricted_gibbs!(i, j, Si, Sj, model, data, assignments)

Gibbs sample assignments for data in `Si` and `Sj`, except for `i` and `j`,
restricting assignments to either cluster `i` or cluster `j`. 
"""
function restricted_gibbs!(i, j, S_full, model, data, assignments)
    zi, zj = assignments[i], assignments[j]

    log_p_transition = 0.0
    ni, nj = 1, 1

    for k in S_full
        if (k == i) || (k == j)
            continue  # Skip this datapoint
        end

        # Remove from current cluster
        xk, zk = data[k], assignments[k]
        remove_datapoint!(model, data[k], zk)

        # Assign to one of two possible clusters
        lp1 = log_posterior_predictive(model.clusters[zi], xk, model)
        lp2 = log_posterior_predictive(model.clusters[zj], xk, model)

        # Sample: log P(C1) = log(exp(lp1) / (exp(lp1) + exp(lp2)) ) = lp1 - logaddexp(lp1, lp2)
        log_p_total = logaddexp(lp1, lp2) 
        log_p_c1 = lp1 - log_p_total
        log_p_c2 = lp2 - log_p_total

        if log(rand()) <= log_p_c1
            # Assign to cluster zi
            add_datapoint!(model, xk, zi)
            log_p_transition += lop_p_c1
            ni += 1
        else
            # Assign to cluster zj
            add_datapoint!(model, xk, zj)
            log_p_transition += lop_p_c2
            nj += 1
        end
    end

    return ni, nj, log_p_transition
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
