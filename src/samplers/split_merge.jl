function split_merge!(
    model::NeymanScottModel, 
    data, 
    assignments;
    num_gibbs=0,
    verbose=false
)
    if num_gibbs > 0
        gibbs_split_merge!(model, data, assignments; num_gibbs=num_gibbs, verbose=verbose)
    end

    i, A = get_split_merge_candidate(model, data, assignments)

    # Quit early if there aren't any neighbors nearby
    if isempty(A)
        return
    end
    j = rand(A)

    if assignments[i] == assignments[j]
        split_move!(i, j, model, data, assignments, verbose)
    else
        merge_move!(i, j, model, data, assignments, verbose)
    end
end

function merge_move!(i, j, model, data, assignments, verbose)
    zi, zj = assignments[i], assignments[j]

    # Compute probability of old partition
    Si = findall(==(zi), assignments)
    Sj = findall(==(zj), assignments)
    n1, n2 = length(Si), length(Sj)

    log_p_old = sm_move_unnormalized_prior(n1, n2, model)
    log_p_old += log_marginal_event(zi, Si, model, data, assignments)
    log_p_old += log_marginal_event(zj, Sj, model, data, assignments)

    # [A] Merge all spikes into the first cluster (zi)
    move_partition!(Sj, model, data, assignments, zj, zi)

    # [B] Compute acceptance probabibility
    # Proposal probability
    log_q_rev = (n1 + n2 - 2) * log(0.5)
    log_q_fwd = log(1.0)

    # Probability of new partition
    log_p_new = sm_move_unnormalized_prior(n1+n2, 0, model)
    log_p_new += log_marginal_event(zi, [Si; Sj], model, data, assignments)

    log_accept_prob = (log_q_rev - log_q_fwd) + (log_p_new - log_p_old)

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        verbose && println("Merge accepted\n")
    else
        # Undo assignments (move back to zj)
        zj = add_cluster!(clusters(model))    
        move_partition!(Sj, model, data, assignments, zi, zj)
    end
end

function split_move!(i, j, model, data, assignments, verbose)
    zi, zj = assignments[i], assignments[j]

    # Compute probability of old partition
    S_full = findall(==(zi), assignments)
    n_full = length(S_full)

    log_p_old = sm_move_unnormalized_prior(n_full, 0, model)
    log_p_old += log_marginal_event(zi, S_full, model, data, assignments)

    # Create a new cluster and move xj over
    zj = add_cluster!(clusters(model))
    assignments[j] = move_datapoint!(model, data[j], zi, zj)

    # [A] Split up remaining spikes
    Si, Sj = split_randomly!(i, j, S_full, model, data, assignments)
    n1, n2 = length(Si), length(Sj)

    # [B] Compute acceptance probabibility
    # Proposal probability
    log_q_fwd = (n1 + n2 - 2) * log(0.5)
    log_q_rev = log(1.0)

    # Probability of new partition
    log_p_new = sm_move_unnormalized_prior(n1, n2, model)
    log_p_new += log_marginal_event(zi, Si, model, data, assignments)
    log_p_new += log_marginal_event(zj, Sj, model, data, assignments)

    log_accept_prob = (log_q_rev - log_q_fwd) + (log_p_new - log_p_old)

    # [C] Accept or reject
    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing        
        verbose && println("Split accepted\n")
    else
        # Undo assignments (move back to zi)
        move_partition!(Sj, model, data, assignments, zj, zi)
    end
end

# ====
# HELPERS
# ====

function get_split_merge_candidate(model, data, assignments)

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

    A = filter!(k -> is_viable_candidate(k, i, data, assignments, window_size), collect(1:n))

    return i, A
end

function is_viable_candidate(k, i, data, assignments, window_size)
    xi, xk = data[i], data[k]
    return (k != i) && (assignments[k] != -1) && (norm(position(xi) - position(xk)) < window_size)
end

function sm_move_unnormalized_prior(n1, n2, model)
    (; α) = model.priors.cluster_amplitude
    mnclp = model.new_cluster_log_prob

    if n2 == 0  # One cluster
        return lgamma(α + n1 + n2) - lgamma(α)
    else  # Two clusters
        return (mnclp - log(α)) + lgamma(α + n1) + lgamma(α + n2) - 2*lgamma(α)
    end
end 

function move_datapoint!(model, x, z1, z2)
    remove_datapoint!(model, x, z1)
    add_datapoint!(model, x, z2)
    return z2
end

function move_partition!(S, model, data, assignments, z1, z2)
    for k in S
        assignments[k] = move_datapoint!(model, data[k], z1, z2)
    end
end

function split_randomly!(i, j, S, model, data, assignments)
    zi, zj = assignments[i], assignments[j]
    Si, Sj = [i], [j]

    for k in S
        if (k == i) || (k == j)
            continue  # Skip
        end

        if rand() < 0.5
            push!(Si, k)
        else
            remove_datapoint!(model, data[k], assignments[k])
            assignments[k] = add_datapoint!(model, data[k], zj)
            push!(Sj, k)
        end
    end

    return Si, Sj
end