VALID_BIRTH_PROPOSALS = [:uniform, :datapoint]

Base.@kwdef struct ReversibleJumpSampler <: AbstractSampler
    verbose::Bool = false
    save_interval::Int = 1
    save_keys::Union{Symbol, Tuple{Vararg{Symbol}}} = (:log_p, :assignments, :clusters, :globals, :time)
    num_samples::Int = 100
    birth_prob::Union{Real, Function} = 0.5
    birth_proposal::Symbol = :uniform
    num_split_merge::Int = 0
    split_merge_gibbs_moves::Int = 0
    num_move::Int = 10
end

function get_birth_prob(S::ReversibleJumpSampler, s::Int)
    if typeof(S.birth_prob) <: Real
        return S.birth_prob
    else
        return S.birth_prob(s)
    end
end

function (S::ReversibleJumpSampler)(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint};
    initial_assignments::Union{Symbol, Vector{Int64}}=:background
)
    @assert S.birth_proposal in VALID_BIRTH_PROPOSALS

    # Grab sampling options.
    verbose, save_interval, num_samples = S.verbose, S.save_interval, S.num_samples

    # Initialize cluster assignments.
    assignments = initialize_assignments(data, initial_assignments)
    recompute_cluster_statistics!(model, clusters(model), data, assignments)

    # Initialize the globals using a custom function and reset model probabilities
    gibbs_initialize_globals!(model, data, assignments)
    _reset_model_probs!(model)

    results = initialize_results(model, assignments, S)
    data_order = collect(1:length(data))

    for s in 1:num_samples
        # [1] Propose a birth death move
        for _ in 1:S.num_move
            birth_death!(model, data; 
                birth_prob=get_birth_prob(S, s), proposal=S.birth_proposal)
        end

        # We need to reset all the assignments and clusters sufficient stats
        assignments .= -1
        # Reset all clusters to empty.
        for k in model.clusters.indices
            reset!(model.clusters[k])
        end

        # [2] Sample parent assignments.        
        for i in data_order
            # Remove i-th datapoint from its current cluster.
            if assignments[i] != -1
                remove_datapoint!(model, data[i], assignments[i]; kill_empty=false)
            else
                remove_bkgd_datapoint!(model, data[i])
            end

            # Sample a new assignment for i-th datapoint.
            assignments[i] = gibbs_conditional_sample_assignment!(model, data[i])
        end

        # [3] Propose a split-merge move
        # recompute_cluster_statistics_in_place!(model, clusters(model), data, assignments)
        for _ in 1:S.num_split_merge
            split_merge!(model, data, assignments; verbose=verbose, num_gibbs=S.split_merge_gibbs_moves)
        end

        # Cluster parameters
        for cluster in clusters(model)
            gibbs_sample_cluster_params!(cluster, model)
        end

        # Global variables
        gibbs_sample_globals!(model, data, assignments)

        # Reset cluster statistics (since global variables have changed)
        recompute_cluster_statistics_in_place!(model, clusters(model), data, assignments)

        # Store results
        if (s % save_interval) == 0
            j = Int(s / save_interval)
            update_results!(results, model, assignments, data, S)
            verbose && print(s, "-")  # Display progress
        end

    end

    verbose && println("Done")

    return results
end

function gibbs_conditional_sample_assignment!(model::NeymanScottModel, x::AbstractDatapoint)
    # Create log-probability vector to sample assignments.
    #  - We need to sample K + 1 possible assignments. We could assign `x` to
    #    one of the K existing clusters. We could also assign `x` to 
    #    the background (index K + 2).

    K = num_clusters(model)
    
    # Grab vector without allocating new memory.
    log_probs = resize!(model.K_buffer, K + 1)

    # Iterate over clusters, indexed by k = {1, 2, ..., K}.
    for (k, cluster) in enumerate(clusters(model))

        # Check if `cluster` is too far away from `k` to be considered.
        if too_far(x, cluster, model) && been_sampled(cluster)
            @debug "Too far!"
            log_probs[k] = -Inf

        else  # Compute probability of adding x to k-th cluster.
            log_probs[k] = log_cluster_intensity(model, cluster, x)
        end
    end

    # Background probability
    log_probs[K + 1] = log_bkgd_intensity(model, x)

    # Sample new assignment for x.
    z = sample_logprobs!(log_probs)

    # Assign datapoint to background.
    if z == (K + 1)
        add_bkgd_datapoint!(model, x)
        return -1

    # Assign datapoint to an existing cluster. There are `K` existing
    # clusters with integer ids held in clusters(model).indices ---
    # we use z as an index into this list.
    else
        k = clusters(model).indices[z]
        add_datapoint!(model, x, k)
        return k
    end
end

function get_empty_clusters(model::NeymanScottModel)
    Cs = clusters(model)
    return [k for k in Cs.indices if Cs[k].datapoint_count == 0]
end

function get_num_empty(model::NeymanScottModel)
    return length(get_empty_clusters(model))
end

function birth_death!(
    model::NeymanScottModel, 
    data::Vector{<: AbstractDatapoint}; 
    birth_prob=0.5, 
    proposal=:uniform
)
    # Compute average space 'occupied' by a single datapoint
    bounds = model.bounds
    σ = 0.1  #maximum(bounds) / cluster_rate(model.priors)
    ℙ_x = MultivariateNormal(length(bounds), σ)

    empty_clusters = get_empty_clusters(model)
    K_empty = get_num_empty(model)
    K_total = num_clusters(model)

    # Propose a birth
    if rand() < birth_prob

        # Make proposal
        k_new = add_cluster!(clusters(model))
        C_new = clusters(model)[k_new]  # <-- ERROR?

        gibbs_sample_cluster_params!(C_new, model)
        if proposal == :datapoint
            x = rand(data)
            C_new.sampled_position = clamp.(x.position + rand(ℙ_x), 0, bounds)
        end

        # Compute acceptance probability
        # p_accept = (p_new / p_old) * (q_death / q_birth) * (1 - birth_prob) / (birth_prob)
        
        ℙ_K = Poisson(cluster_rate(model.priors))
        ℙ_A = cluster_amplitude(model.priors)
        
        # p(X ∪ C_new)
        log_p_accept = (
            logpdf(ℙ_K, K_total+1) 
            + logpdf(ℙ_A, C_new.sampled_amplitude)
            - log(volume(model))
        )

        # 1 / p(X)
        log_p_accept -= logpdf(ℙ_K, K_total)

        # q_death(C_new ; X ∪ C_new)  <-- Uniform death proposals
        log_p_accept += -log(K_total + 1)

        # 1 / q_birth(C_new ; X)
        # q_birth draws cluster marks ϕ uniformly from the prior
        # So since p_new(ϕ | ...) = q_birth(ϕ | ...), these terms cancel
        # Only the amplitude and position terms need to be removed
        log_q_birth = logpdf(posterior(0, ℙ_A), C_new.sampled_amplitude)
        
        if proposal == :uniform
            log_q_birth += -log(volume(model))
        elseif proposal == :datapoint
            position_logp = -Inf
            for x in data
                position_logp = logaddexp(
                    position_logp, 
                    -log(length(data)) + logpdf(ℙ_x, C_new.sampled_position - x.position)
                )
            end
            log_q_birth += position_logp
        else
            error("Invalid birth proposal.")
        end
        log_p_accept -= log_q_birth

        # (1 - birth_prob) / (birth_prob)
        log_p_accept += log(1 - birth_prob) - log(birth_prob)

        #@show "Birth", log_p_accept

        # Accept or reject
        # If rejected, remove the cluster
        if log(rand()) < log_p_accept
            # Accept the cluster!
            #@show "Birth accepted"
            return k_new
        else
            remove_cluster!(clusters(model), k_new)
            return -1
        end

    # Propose a death
    elseif K_empty > 0

        # Make proposal
        k_death = rand(empty_clusters)
        C_death = clusters(model)[k_death]

        # Compute acceptance probability
        # p_accept = (p_new / p_old) * (q_birth / q_death) * (1 - birth_prob) / (birth_prob)
        
        ℙ_K = Poisson(cluster_rate(model.priors))
        ℙ_A = cluster_amplitude(model.priors)
        
        # p(X \ C_new)
        log_p_accept = logpdf(ℙ_K, K_total - 1)
        
        # 1 / p(X)
        log_p_accept -= (
            logpdf(ℙ_K, K_total) 
            + logpdf(ℙ_A, C_death.sampled_amplitude)
            - log(volume(model))
        )

        # q_birth(C_death ; X)
        # q_birth draws cluster marks ϕ uniformly from the prior
        # So since p_new(ϕ | ...) = q_birth(ϕ | ...), these terms cancel
        # Only the amplitude and position terms needs to be removed
        log_q_birth = logpdf(posterior(0, ℙ_A), C_death.sampled_amplitude)
        
        if proposal == :uniform
            log_q_birth += -log(volume(model))
        elseif proposal == :datapoint
            position_logp = -Inf
            for x in data
                position_logp = logaddexp(
                    position_logp, 
                    -log(length(data)) + logpdf(ℙ_x, C_death.sampled_position - x.position)
                )
            end
            log_q_birth += position_logp
        else
            error("Invalid birth proposal.")
        end
        log_p_accept += log_q_birth

        # 1 / q_death(C_new ; X ∪ C_new)  <-- Uniform death proposals
        log_p_accept -= -log(K_total)

        # (birth_prob) / (1 - birth_prob)
        log_p_accept += log(birth_prob) - log(1 - birth_prob)

        #@show "Death", log_p_accept

        # Accept or reject
        # If accepted, remove the cluster
        if log(rand()) < log_p_accept
            #@show "Death accepted."
            remove_cluster!(clusters(model), k_death)
            return k_death
        else
            return -1
        end
    end
end