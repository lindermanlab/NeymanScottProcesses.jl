function split_data(num_partitions, max_time, data, mode)
    # Split into `num_partitions` even intervals
    if mode === :simple
        return Tuple(p * max_time / num_partitions for p in 0:num_partitions)

    # Split so that each partition has a similar number of spikes
    elseif mode === :balanced
        sorted_data = sort(data, by=p->first_coordinate(p))
        points_per_partition = length(data) / num_partitions

        splits = Tuple(
            first_coordinate(sorted_data[ceil(Int, points_per_partition*p)]) + eps()
            for p in 1:num_partitions
        )

        return (0.0, splits...)

    else
        error("Split mode not supported.")
    end
end

"""
Run gibbs sampler.
"""
function gibbs_sample!(
    model::DistributedNeymanScottModel,
    data::Vector{<: AbstractDatapoint},
    initial_assignments::Vector{Int64},
    num_samples::Int64,
    extra_split_merge_moves::Int64,
    split_merge_window::Float64,
    save_every::Int64;
    verbose=false,
    split_mode=:balanced,
    save_set=[:latents, :globals, :assignments],
)

    if extra_split_merge_moves > 0
        @warn "Split merge not implemented for distributed model."
    end

    num_partitions = model.num_partitions

    max_time = first_bound(model)

    # Partition data across first dimension.
    split_points = split_data(num_partitions, max_time, data, split_mode)
    spk_partition = Tuple(AbstractDatapoint[] for m in model.submodels)
    assgn_partition = [Int64[] for m in model.submodels]  # TODO -- Tuple?
    partition_ids = [Int64[] for m in model.submodels]    # TODO -- Tuple?

    for s = 1:length(data)
        for p = 1:model.num_partitions
            if split_points[p] <= first_coordinate(data[s]) <= split_points[p + 1]
                push!(spk_partition[p], data[s])
                push!(assgn_partition[p], initial_assignments[s])
                push!(partition_ids[p], s)
                break
            end
        end
    end

    verbose && println("Split points: ", split_points)
    verbose && println("Partition sizes: ", length.(spk_partition))

    # Dense rank assignments within each partition.
    for p = 1:model.num_partitions
        idx = assgn_partition[p] .> 0 # ignore background data.
        assgn_partition[p][idx] .= denserank(assgn_partition[p][idx])
    end

    # Pass assignments to submodels
    for p in 1:model.num_partitions
        recompute!(
            model.submodels[p],
            spk_partition[p],
            assgn_partition[p],
        )
    end

    # Save datapoint assignments over samples.
    assignments = initial_assignments
    collect_assignments!(model, assignments, assgn_partition, partition_ids)

    # Order to iterate over datapoints.
    spike_order_partition = Tuple(
        collect(1:length(subdata)) for subdata in spk_partition
    )

    # ======== THINGS TO SAVE ======== #
    n_saved_samples = Int(round(num_samples / save_every))
    log_p_hist = zeros(n_saved_samples)
    assignment_hist = zeros(Int64, length(data), n_saved_samples)
    latent_event_hist = Vector{Any}[]
    globals_hist = AbstractGlobals[]

    # ======== MAIN LOOP ======== #

    # Draw samples.
    for s = 1:num_samples

        # Update spike assignments in random order.
        Threads.@threads for p in 1:model.num_partitions
            Random.shuffle!(spike_order_partition[p])
            for i in spike_order_partition[p]
                if assgn_partition[p][i] != -1
                    remove_datapoint!(
                        model.submodels[p], 
                        spk_partition[p][i], 
                        assgn_partition[p][i]
                    )
                end
                assgn_partition[p][i] = gibbs_add_datapoint!(
                    model.submodels[p], 
                    spk_partition[p][i]
                )
            end
        end

        # Pass spike assignment updates and latent event updates to primary model.
        # Latent events.
        collect_assignments!(model, assignments, assgn_partition, partition_ids)

        # Update latent events.
        gibbs_update_latents!(model.primary_model)

        # Update globals
        gibbs_update_globals!(model.primary_model, data, assignments)

        # Make sure model probabilities are correct after updating globals
        # Note that this (redundantly) sets the model probabilities for both the 
        # primary model and the submodels
        _gibbs_reset_model_probs(model)

        # Recompute sufficient statistics
        recompute!(model.primary_model, data, assignments)

        # No need to pass updates to submodels, since the globals
        # and events point to the same objects.

        # Store results
        if (s % save_every) == 0
            j = Int(s / save_every)
            save_sample!(
                log_p_hist, assignment_hist, latent_event_hist, globals_hist,
                j, model.primary_model, data, assignments, save_set
            )
            verbose && print(s, "-")
        end
    end
    verbose && println("Done")

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )
end


"""
Collects assignments from submodels into `assignments` vector
and populates `events(model.primary_model)` with
submodel's events.
"""
function collect_assignments!(
        model::DistributedNeymanScottModel,
        assignments::Vector{Int64},
        assgn_partition::Vector{Vector{Int64}},
        partition_ids::Vector{Vector{Int64}}
    )

    empty!(events(model.primary_model).events)
    empty!(events(model.primary_model).indices)
    event_id = 1
    event_map = Dict{Tuple{Int64,Int64},Int64}()

    # Create flat assignment index for each event, to be used
    # in `primary_model`.
    for p = 1:model.num_partitions

        # Iterate over event indices in submodel-p.
        for ind in events(model.submodels[p]).indices
            event_map[(p, ind)] = event_id
            push!(
                events(model.primary_model).events,
                events(model.submodels[p])[ind]
            )
            push!(
                events(model.primary_model).indices,
                event_id
            )
            event_id += 1
        end

        # Assign each spike in `primary_model` using the
        # event ids generated above.
        for (s, assgn) in enumerate(assgn_partition[p])
            if assgn == -1
                assignments[partition_ids[p][s]] = -1
            else
                assignments[partition_ids[p][s]] = event_map[(p, assgn)]
            end
        end

    end
end


function _gibbs_reset_model_probs(model::DistributedNeymanScottModel)
    _gibbs_reset_model_probs(model.primary_model)
    for submodel in model.submodels
        _gibbs_reset_model_probs(submodel)
    end
end
