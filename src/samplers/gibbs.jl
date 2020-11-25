struct GibbsSampler
    verbose::Bool
    save_interval::Int
    save_set::Vector{Symbol}
    num_samples::Int
end

initialize_globals!(model, data, assignments) = notimplemented()
remove_bkgd_datapoint!(model, x) = notimplemented()

function (S::GibbsSampler)(
    model::AbstractModel, 
    data::Vector{<: AbstractDatapoint}, 
    initial_assignments::Vector{Int64},
)
    verbose, save_interval, num_samples = S.verbose, S.save_interval, S.num_samples

    # Initialize spike assignments.
    assignments = deepcopy(initial_assignments)
    recompute_statistics!(model, data, assignments)

    # Initialize the globals using a custom function and reset model probabilities
    initialize_globals!(model, data, assignments)
    _reset_model_probs(model)

    results = initialize_results(model, assignments, S)
    spike_order = collect(1:length(data))

    for s in 1:num_samples

        Random.shuffle!(spike_order)  # Update spike assignments in random order.
        for i in spike_order

            if assignments[i] != -1
                remove_datapoint!(model, data[i], assignments[i])
            else
                remove_bkgd_datapoint!(model, data[i])
            end

            assignments[i] = gibbs_add_datapoint!(model, data[i], S.H)
        end

        # Update latent events and global variables
        gibbs_update_latents!(model)
        gibbs_update_globals!(model, data, assignments)

        _reset_model_probs(model)  # Recompute background and new cluster probabilities
        recompute_statistics!(model, data, assignments)  # Recompute event statistics

        # Store results
        if (s % save_interval) == 0
            j = Int(s / save_interval)
            update_results!(results, model, assignments, data, S)
            verbose && print(s, "-")  # Display progress
        end
    end
    verbose && println("Done") # Finish progress bar.

    return results
end