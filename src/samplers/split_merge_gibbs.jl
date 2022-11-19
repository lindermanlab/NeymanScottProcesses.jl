function gibbs_split_merge!(
    model::NeymanScottModel, 
    data, 
    assignments;
    num_gibbs=5,
    verbose=false
)
    i, A = get_split_merge_candidate(model, data, assignments)

    # Quit early if there aren't any neighbors nearby
    if isempty(A)
        return
    end
    j = rand(A)

    # Store the initial state, so we can go back if needed
    Si0, Sj0, log_prob_old = get_initial_split_move_state(i, j, assignments)
    S_full = [Si0; Sj0]

    # Create launch state
    # We first partition the data into Si, Sj randomly
    # NOTE: This mutates both `model` and `assignments`
    Si, Sj = create_launch_state!(i, j, S_full, model, data, assignments)

    move = (i, j, Si, Sj, S_full)
    initial_state = (Si0, Sj0, log_prob_old)
 
    if assignments[i] == assignments[j]
        gibbs_split_move!(move, initial_state, data, assignments, verbose)
    else
        gibbs_merge_move!(move, initial_state, data, assignments, verbose)
    end
end

function gibbs_merge_move!(move, initial_state, data, assignments, verbose)
    (i, j, Si, Sj, S_full) = move
    (Si0, Sj0, log_p_old) = initial_state

    zi, zj = assignments[i], assignments[j]

    # [A] Get proposal probabilities
    log_q_rev = gibbs_transition_probability(i, j, Si, Sj, Si0, Sj0, model, data, assignments)
    log_q_fwd = log(1.0)

    # [B] Merge all spikes in Sj into the first cluster Si
    move_partition!(Sj, model, data, assignments, zj, zi)

    # Probability of new partition
    n_full = length(S_full)
    log_p_new = sm_move_unnormalized_prior(n_full, 0, model)
    log_p_new += log_marginal_event(zi, S_full, model, data, assignments)

    # [C] Accept or reject
    log_accept_prob = (log_q_rev - log_q_fwd) + (log_p_new - log_p_old)

    if log(rand()) < log_accept_prob
        # Leave the new assignments as is, do nothing
        verbose && println("Merge accepted\n")
    else
        # Undo assignments (move back to Sj0)
        zj = add_cluster!(clusters(model))    
        move_partition!(Sj0, model, data, assignments, zi, zj)
    end
end

function gibbs_split_move!(move, initial_state, data, assignments, verbose)
    (i, j, Si, Sj, S_full) = move
    (Si0, Sj0, log_p_old) = initial_state

    zi, zj = assignments[i], assignments[j]

    # [A] Propose split state
    log_p_transition = restricted_gibbs!(i, j, S_full, model, data, assignments)
    Si, Sj = filter(==(zi), assignments), filter(==(zj), assignments)
    n1, n2 = length(S1), length(S2)

    # [B] Compute acceptance probabibility
    # Proposal probability
    log_q_rev = log(1.0)
    log_q_fwd = log_p_transition

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

"""
    gibbs_transition_probability(i, j, Si, Sj, Si0, Sj0, model, data, assignments)

Calculate the Gibbs transition probability from moving from launch state (Si, Sj) 
to other state (Si0, Sj0).
"""
function gibbs_transition_probability(i, j, Si, Sj, Si0, Sj0, model, data, assignments)
    zi, zj = assignments[i], assignments[j]

    log_p_transition = 0.0

    for k in Iterators.flatten((Si, Sj))
        if (k == i) || (k == j)
            continue  # Skip this datapoint
        end

        # Remove from current cluster
        xk, zk = data[k], assignments[k]
        remove_datapoint!(model, data[k], zk)

        # Likelihood of two possible clusters
        lp1 = log_posterior_predictive(model.clusters[zi], xk, model)
        lp2 = log_posterior_predictive(model.clusters[zj], xk, model)

        # Probabilities: log P(C1) = log(exp(lp1) / (exp(lp1) + exp(lp2)) ) = lp1 - logaddexp(lp1, lp2)
        log_p_total = logaddexp(lp1, lp2) 
        log_p_c1 = lp1 - log_p_total
        log_p_c2 = lp2 - log_p_total

        if k in Si0
            # Assign to cluster zi
            add_datapoint!(model, xk, zi)
            log_p_transition += lop_p_c1
        else
            # Assign to cluster zj
            add_datapoint!(model, xk, zj)
            log_p_transition += lop_p_c2
        end
    end

    return log_p_transition
end

"""
    restricted_gibbs!(i, j, Si, Sj, model, data, assignments)

Gibbs sample assignments for data in `Si` and `Sj`, except for `i` and `j`,
restricting assignments to either cluster `i` or cluster `j`. 
"""
function restricted_gibbs!(i, j, S_full, model, data, assignments)
    zi, zj = assignments[i], assignments[j]

    log_p_transition = 0.0

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
        else
            # Assign to cluster zj
            add_datapoint!(model, xk, zj)
            log_p_transition += lop_p_c2
        end
    end

    return log_p_transition
end

function create_launch_state!(i, j, S, model, data, assignments)
    zi, zj = assignments[i], assignments[j]

    # Create a new cluster if needed
    if assignments[i] == assignments[j]
        zj = add_cluster!(clusters(model))
        assignments[j] = move_datapoint!(model, data[j], zi, zj)
    end

    # Initialize launch state
    split_randomly!(i, j, S, model, data, assignments)

    # Then apply `t` rounds of Gibbs sampling
    for _ in 1:num_gibbs
        restricted_gibbs!(i, j, S, model, data, assignments)
    end
    
    Si = filter(==(zi), assignments)
    Sj = filter(==(zj), assignments)

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