"""
Constructs PPSeq object from config dict.
"""
function construct_model(config::Dict,
                         max_time::Float64,
                         num_neurons::Int64)

    # Prior on sequence type proportions / relative frequencies.
    seq_type_proportions = SymmetricDirichlet(
        config[:seq_type_conc_param],
        config[:num_sequence_types]
    )

    # Prior on expected number of spikes induces by a sequence events.
    seq_event_amplitude = specify_gamma(
        config[:mean_event_amplitude],    # mean of gamma; α / β
        config[:var_event_amplitude]      # variance of gamma; α / β²
    )

    # Prior on relative response amplitudes per neuron to each sequence type.
    neuron_response_proportions = SymmetricDirichlet(
        config[:neuron_response_conc_param],
        num_neurons
    )

    # Prior on the response offsets and widths for each neuron.
    neuron_response_profile = NormalInvChisq(
        config[:neuron_offset_pseudo_obs],
        config[:neuron_offset_prior],
        config[:neuron_width_pseudo_obs],
        config[:neuron_width_prior],
    )

    # Prior on expected number of background spikes in a unit time interval.
    bkgd_amplitude = specify_gamma(   
        config[:mean_bkgd_spike_rate],    # mean of gamma; α / β
        config[:var_bkgd_spike_rate]      # variance of gamma; α / β²
    )

    # Prior on relative background firing rates across neurons.
    bkgd_proportions = SymmetricDirichlet(
        config[:bkgd_spikes_conc_param],
        num_neurons
    )

    PPSeq(
        # constants
        max_time,
        config[:max_sequence_length],

        # warp parameters
        config[:num_warp_values],
        config[:max_warp],
        config[:warp_variance],

        # priors
        config[:seq_event_rate],
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )
end


"""
Trains PPSeq model given config dict.
"""
function train!(
        model::PPSeq,
        spikes::Vector{Spike},
        initial_assignments::Vector{Int64},
        config::Dict;
        verbose=true
    )

    # Save copy of initial assignments.
    _inits = copy(initial_assignments)

    # Set random seed.
    Random.seed!(config[:random_seed])

    # Set up parallel MCMC, if requested.
    if config[:num_threads] >= 1
        training_model = DistributedPPSeq(model, config[:num_threads])
    else
        training_model = model
    end
    @show typeof(training_model)

    # Draw annealed Gibbs samples.
    (
        assignments,
        anneal_assignment_hist,
        anneal_log_p_hist,
        anneal_latent_event_hist,
        anneal_globals_hist
    ) =
    PointProcessSequences.annealed_gibbs!(
        training_model,
        spikes,
        initial_assignments,
        config[:num_anneals],
        config[:samples_per_anneal],
        config[:max_temperature],
        config[:split_merge_moves_during_anneal],
        config[:split_merge_window],
        config[:save_every_during_anneal];
        verbose=verbose
    )

    # Sanity check.
    for k in model.sequence_events.indices
        event = model.sequence_events[k]
        @assert event.spike_count == sum(assignments .== k)
    end

    # Draw regular Gibbs samples.
    (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    gibbs_sample!(
        training_model,
        spikes,
        assignments,
        config[:samples_after_anneal],
        config[:split_merge_moves_after_anneal],
        config[:split_merge_window],
        config[:save_every_after_anneal];
        verbose=verbose
    )

    # Sanity check.
    for k in model.sequence_events.indices
        event = model.sequence_events[k]
        @assert event.spike_count == sum(assignments .== k)
    end

    return Dict(
        
        # Initial assignment variables.
        :initial_assignments => _inits,

        # Results during annealing.
        :anneal_assignment_hist => anneal_assignment_hist,
        :anneal_log_p_hist => anneal_log_p_hist,
        :anneal_latent_event_hist => anneal_latent_event_hist,
        :anneal_globals_hist => anneal_globals_hist,

        # Results after annealing.
        :final_assignments => assignments,
        :assignment_hist => assignment_hist,
        :latent_event_hist => latent_event_hist,
        :globals_hist => globals_hist,
        :log_p_hist => log_p_hist
    )

end


"""
Trains PPSeq model with masked data.
"""
function train!(
        model::PPSeq,
        spikes::Vector{Spike},
        masks::Vector{Mask},
        initial_assignments::Vector{Int64},
        config::Dict
    )

    # If masks are empty then call regular training function.
    isempty(masks) && return train!(model, spikes, initial_assignments, config)

    # Save copy of initial assignments.
    _inits = copy(initial_assignments)

    # Set random seed.
    Random.seed!(config[:random_seed])

    # Set up parallel MCMC, if requested.
    if config[:num_threads] > 1
        training_model = DistributedPPSeq(model, config[:num_threads])
    else
        training_model = model
    end

    # # Sanity check.
    # for k in model.sequence_events.indices
    #     event = model.sequence_events[k]
    #     @assert event.spike_count == sum(initial_assignments .== k)
    # end

    # Draw annealed Gibbs samples.
    (
        assignments,
        anneal_assignment_hist,
        anneal_train_log_p_hist,
        anneal_test_log_p_hist,
        anneal_latent_event_hist,
        anneal_globals_hist
    ) =
    annealed_masked_gibbs!(
        training_model,
        spikes,
        masks,
        initial_assignments,
        config[:num_anneals],
        config[:max_temperature],
        config[:num_spike_resamples_per_anneal],
        config[:samples_per_resample],
        config[:split_merge_moves_during_anneal],
        config[:split_merge_window],
        config[:save_every_during_anneal];
        verbose=true
    )

    # # Sanity check.
    # for k in model.sequence_events.indices
    #     event = model.sequence_events[k]
    #     @assert event.spike_count == sum(assignments .== k)
    # end

    # Draw regular Gibbs samples.
    (
        assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    masked_gibbs!(
        training_model,
        spikes,
        masks,
        assignments,
        config[:num_spike_resamples_after_anneal],
        config[:samples_per_resample],
        config[:split_merge_moves_after_anneal],
        config[:split_merge_window],
        config[:save_every_after_anneal];
        verbose=true
    )

    # # Sanity check.
    # for k in model.sequence_events.indices
    #     event = model.sequence_events[k]
    #     @assert event.spike_count == sum(assignments .== k)
    # end

    return Dict(

        :initial_assignments => _inits,

        # Results during annealing.
        :anneal_assignment_hist => anneal_assignment_hist,
        :anneal_train_log_p_hist => anneal_train_log_p_hist,
        :anneal_test_log_p_hist => anneal_test_log_p_hist,
        :anneal_latent_event_hist => anneal_latent_event_hist,
        :anneal_globals_hist => anneal_globals_hist,

        # Results after annealing.
        :final_assignments => assignments,
        :assignment_hist => assignment_hist,
        :latent_event_hist => latent_event_hist,
        :globals_hist => globals_hist,
        :train_log_p_hist => train_log_p_hist,
        :test_log_p_hist => test_log_p_hist,
        :masks => masks
    )

end

"""
Loads data, trains model, saves results and config dict file.
"""
function load_train_save(config::Dict)

    # Load data.
    spikes, max_time, num_neurons = load_dataset(config[:dataset_id])

    # Construct model.
    model = construct_model(config, max_time, num_neurons)

    # Mask data for cross-validation.
    not_masking = (
       !(:mask_data in keys(config)) ||
       !(config[:mask_data] in ("randomly", "in blocks"))
    )

    if not_masking

        # No spikes are masked.
        println("Fitting parameters without masks...")
        masks = Mask[]
        initial_assignments = initialize!(
            model,
            spikes,
            config
        )

    else

        # Random masking.
        if config[:mask_data] == "randomly"

            println("Creating random mask...")
            Random.seed!(config[:random_seed])
            masks = create_random_mask(
                num_neurons,
                model.max_time,
                config[:mask_lengths],
                config[:percent_masked]
            )

        # Blocked masking (less optimal, but easy to visualize).
        elseif config[:mask_data] == "in blocks"

            println("Creating blocked mask...")
            Random.seed!(config[:random_seed])
            masks = create_blocked_mask(
                num_neurons,
                model.max_time
            )

        else
            throw(ArgumentError("Mask configuration not recognized"))
        end

        # Remove masked spikes.
        _, unmasked_spikes =  split_spikes_by_mask(spikes, masks)

        # Sample new spikes in masked region.
        sampled_spikes, _ = sample_masked_spikes!(Spike[], Int64[], model, masks)

        # Initialize using imputed spikes.
        initial_assignments = initialize!(
            model,
            vcat(unmasked_spikes, sampled_spikes),
            config
        )

    end

    # Save a copy of the initial model.
    initial_model = deepcopy(model)

    # Train model. If masks is empty, then regular sampling script is used.
    results = train!(model, spikes, masks, initial_assignments, config)

    # Construct directory to output results.
    jobpath = get_jobpath(
        config[:dataset_id],
        config[:sweep_id],
        config[:job_id]
    )
    mkpath(jobpath)

    # Save config file.
    YAML.write_file(joinpath(jobpath, "config.yml"), config)

    # Save model.
    BSON.@save joinpath(jobpath, "model.bson") model
    BSON.@save joinpath(jobpath, "initial_model.bson") initial_model

    # Save assignments and log likelihood history.
    BSON.bson(joinpath(jobpath, "results.bson"), results)
    return results

end


function initialize!(
        model::PPSeq,
        spikes::Vector{Spike},
        config::Dict
    )

    if config[:initialization] == "background"
        return fill(-1, length(spikes))

    elseif config[:initialization] == "debug"
        assignments = BSON.load("./data/hippocampus/maze_good_spike_assignments.bson")[:assignments]
        PointProcessSequences.recompute!(model, spikes, assignments)
        for k = 1:1000
            PointProcessSequences.gibbs_update_latents!(model)
            PointProcessSequences.gibbs_update_globals!(model, spikes, assignments)
        end
        return assignments

    elseif config[:initialization] == "convnmf"

        # Now fit convnmf for initialization.
        binsize = config[:convnmf_binsize]
        T = model.max_time
        N = PointProcessSequences.num_neurons(model)
        R = config[:num_sequence_types]
        datamat = zeros(N, ceil(Int, T / binsize))

        for spk in spikes
            timebin = ceil(Int, eps() + spk.timestamp / binsize)
            datamat[spk.neuron, timebin] += 1
        end

        # Fit convNMF model.
        print("Fitting convNMF model to initialize pp-seq model....")
        cmf = CMF.fit_cnmf(
            datamat,
            L=floor(Int, config[:convnmf_seqlen] / binsize),
            K=R,
            alg=CMF.PGDUpdate,
            max_itr=200,
            max_time=Inf,
            constrW=CMF.NonnegConstraint(),
            constrH=CMF.NonnegConstraint(),
            penaltiesW=[CMF.SquarePenalty(1)],
            penaltiesH=[CMF.AbsolutePenalty(1)],
            check_convergence=false,
            seed=config[:random_seed]
        )
        println("Done!")

        # Find peaks corresponding to latent events.
        n_latents = 0
        event_ids = Vector{Int64}[]
        peak_bins = Vector{Int64}[]
        for r = 1:R
            pb, _ = Peaks.peakprom(
                cmf.H[r, :],
                Peaks.Maxima(), 
                floor(Int, .5 * config[:convnmf_seqlen] / binsize),
                config[:convnmf_peak_thres],
            )
            push!(peak_bins, pb)
            push!(event_ids, range(1 + n_latents, step=1, length=length(pb)))
            n_latents += length(pb)
        end

        reconstructions = Tuple(CMF.tensor_conv(cmf.W[r:r, :, :], cmf.H[r:r, :]) for r=1:R) 
        thresholds = config[:convnmf_bkgd_thres] .* dropdims(maximum(sum(reconstructions), dims=2), dims=2)

        assignments = fill(-1, length(spikes))
        for (i, spk) in enumerate(spikes)
            timebin = ceil(Int, eps() + spk.timestamp / binsize)
            seqtype = argmax([recon[spk.neuron, timebin] for recon in reconstructions])
            if reconstructions[seqtype][spk.neuron, timebin] > thresholds[spk.neuron]
                assignments[i] = event_ids[seqtype][argmin(abs.(peak_bins[seqtype] .- timebin))]
            end
            # else, keep neuron in background.
        end

        # Do some iterations to fit the global variables.
        PointProcessSequences.recompute!(model, spikes, assignments)
        PointProcessSequences.gibbs_update_latents!(model)
        PointProcessSequences.gibbs_update_globals!(model, spikes, assignments)
        for k = 1:1000
            PointProcessSequences.gibbs_update_latents!(model)
            PointProcessSequences.gibbs_update_globals!(model, spikes, assignments)
        end

        return assignments

    else
        throw(ArgumentError("Did not recognize initialization method."))
    end

end


"""
Heuristically merges latent events close together. Can be useful
to develop some ad hoc initializations.
"""
function heuristic_merge!(
        model::PPSeq,
        spikes::Vector{Spike},
        assignments::Vector{Int64},
    )

    idx = model.sequence_events.indices
    taus = [event.sampled_timestamp for event in model.sequence_events]
    rs = [event.sampled_type for event in model.sequence_events]

    D = abs.(reshape(taus, length(taus), 1) .- reshape(taus, 1, length(taus)))
    F = abs.(reshape(rs, length(rs), 1) .- reshape(rs, 1, length(rs)))

    D[F .!= 0] .= Inf
    D[diagind(D)] .= Inf

    k, j = [idx[_i] for _i in Tuple(argmin(D))]

    if minimum(D) > 0.5
        println("Events too far to merge.")
        return assignments
    end

    println(idx)

    println("merging ", k, " into ", j, "...")

    scj = model.sequence_events[j].spike_count
    sck = model.sequence_events[k].spike_count
    nev = length(model.sequence_events)

    z = 0

    for i = 1:length(spikes)
        if assignments[i] == k
            remove_datapoint!(model, spikes[i], k)
            assignments[i] = add_datapoint!(model, spikes[i], j)
            z += 1
        end
    end

    return assignments

end


function train_convnmf(
    spikes::Vector{Spike},
    config
)
    binsize = config[:convnmf_binsize]
    T = config[:max_time]
    N = config[:num_neurons]
    R = config[:num_sequence_types]
    datamat = zeros(N, ceil(Int, T / binsize))

    for spk in spikes
        timebin = ceil(Int, eps() + spk.timestamp / binsize)
        datamat[spk.neuron, timebin] += 1
    end

    # Fit convNMF model.
    cmf = CMF.fit_cnmf(
        datamat,
        L=floor(Int, config[:convnmf_seqlen] / binsize),
        K=R,
        alg=CMF.PGDUpdate,
        max_itr=200,
        max_time=Inf,
        constrW=CMF.NonnegConstraint(),
        constrH=CMF.NonnegConstraint(),
        penaltiesW=[CMF.SquarePenalty(1)],
        penaltiesH=[CMF.AbsolutePenalty(1)],
        check_convergence=false,
        seed=config[:random_seed]
    )
    println("Done!")

    return cmf
end