using Interpolations

# Loads PPSeq module, and file IO wrappers.
print("Loading utils...")
include("./utils/all_utils.jl")
println("Done!")

config = Dict{Symbol,Union{Int64,Float64,String}}(
    :seq_type_conc_param => 3.0,
    :num_sequence_types =>  2,
    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,
    :neuron_response_conc_param => 1.0,
    :neuron_offset_pseudo_obs => 0.15,
    :neuron_offset_prior => 0.0,
    :neuron_width_pseudo_obs => 4.0,
    :neuron_width_prior => 0.1,
    :num_warp_values => 1,
    :warp_variance => 1.0,
    :mean_bkgd_spike_rate => 500.0,
    :var_bkgd_spike_rate => 200.0,
    :bkgd_spikes_conc_param => 1.0,
    :max_sequence_length => 30.0,
    :seq_event_rate => .001,
    :dataset_id => "big_maze",
    :random_seed => 1234,
    :num_threads => 4,
    :max_warp => 1.0,
    :num_anneals => 0,
    :samples_per_anneal => 100,
    :max_temperature => 100.0,
    :samples_per_resample => 1,
    :num_spike_resamples_per_anneal => 100,
    :num_spike_resamples_after_anneal => 300,
    :save_every_during_anneal => 10,
    :samples_after_anneal => 50,
    :save_every_after_anneal => 1,
    :split_merge_moves_during_anneal => 100,
    :split_merge_moves_after_anneal => 1000,
    :split_merge_window => 5.0,
    :initialization => "convnmf",
    :convnmf_seqlen => 10.0,
    :convnmf_binsize => 1.0,
    :convnmf_peak_thres => 0.5,
    :convnmf_bkgd_thres => 0.2,
    :mask_data => "none",
    :mask_lengths => 30.0,
    :percent_masked => 5.0,
)


for num_threads in (4, 1)

    # Set num threads.
    config[:num_threads] = num_threads
    @show config[:num_threads]

    # Load data, construct model, initialize assignments.
    print("Setting up model...")
    spikes, max_time, num_neurons = load_dataset(config[:dataset_id])
    @show length(spikes)
    @show num_neurons
    @show max_time
    model = construct_model(config, max_time, num_neurons)
    assignments = initialize!(
        model,
        spikes,
        config
    )
    println("DONE!")

    # Set up parallel MCMC, if requested.
    if config[:num_threads] >= 1
        training_model = DistributedPPSeq(model, config[:num_threads])
    else
        training_model = model
    end
    @show typeof(training_model)

    # Run performance test
    log_p_hist = Float64[]
    num_epochs = 10
    save_every = 10
    epoch_length = 100
    times = Float64[]

    for epoch = 1:num_epochs
        @printf("Epoch : %i / %i\n", epoch, num_epochs)

        # Draw regular Gibbs samples.
        t = @elapsed (
            assignments, # re-used on next iteration
            assignment_hist, # ignored
            lps,
            latent_event_hist, # ignored
            globals_hist # ignored
        ) =
        gibbs_sample!(
            training_model,
            spikes,
            assignments,
            epoch_length,
            0, # No split merge moves for performance
            0.0,
            save_every; # save log-likelihood every 10 samples
            verbose=false
        )
        push!(times, t)
        append!(log_p_hist, lps)
        @printf(" elapsed = %f\n", t)
        (sum(times) > 3600) && break
    end

    interp = LinearInterpolation(
        range(1, length(log_p_hist), length=length(times)),
        cumsum(times)
    )
    interp_times = interp(collect(1:length(log_p_hist)))

    BSON.@save @sprintf("./figures/performance/big_maze_perf_nthread_%i.bson", config[:num_threads]) log_p_hist interp_times

end
