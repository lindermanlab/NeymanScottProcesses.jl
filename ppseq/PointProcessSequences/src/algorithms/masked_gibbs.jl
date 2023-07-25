function masked_proportion(model::NeymanScottModel, masks::Vector{<: AbstractMask})
    masked_area = 0.0
    for mask in masks
        masked_area += volume(mask)
    end
    return masked_area / volume(model)
end
function masked_proportion(model::PPSeq, masks::Vector{SeqMask})
    masked_area = 0.0
    for mask in masks
        masked_area += volume(mask)
    end
    return masked_area / (volume(model) * num_neurons(model))
end


function sample_background(model::PPSeq)
    num_samples = rand(Poisson(bkgd_rate(globals(model) * volume(model))))
    bkgd_dist = Categorical(exp.(globals(model).bkgd_log_proportions))

    neurons = rand(bkgd_dist, num_samples)
    times = rand(num_samples) * model.max_time

    return Spike.(neurons, times)
end
function sample_background(model::GaussianNeymanScottModel)
    num_samples = rand(Poisson(bkgd_rate(globals(model)) * volume(model)))

    samples = Point[]
    for s in 1:num_samples
        push!(samples, Point(rand(2) .* bounds(model)))
    end

    return samples
end
function sample_background(model::CablesModel) notimplemented() end


function sample(event::SeqEvent, model::PPSeq) 
    globals = globals(model)

    # Compute neuron distributions
    neuron_rel_amps = exp.(globals.neuron_response_log_proportions)
    neuron_dists = [
        Categorical(neuron_rel_amps[:, i]) 
        for i = 1:size(neuron_rel_amps, 2)
    ]

    # Number of spikes evoked by latent event.
    num_samples = rand(Poisson(event.sampled_amplitude))

    # Sample neuron, then spike time, for each spike.
    samples = Spike[]
    for n in rand(neuron_dists[event.sampled_type], S) # TODO

        # Sample spike time.
        μ = globals.neuron_response_offsets[n, event.sampled_type]
        σ = sqrt(globals.neuron_response_widths[n, event.sampled_type])
        w = model.priors.warp_values[event.sampled_warp]
        t = w * (σ * randn() + μ) + event.sampled_timestamp

        push!(samples, Spike(n, t))
    end

    return samples
end
function sample(event::Cluster, model::GaussianNeymanScottModel) 
    num_samples = rand(Poisson(event.sampled_amplitude))
    distr = MultivariateNormal(position(event), covariance(event))

    samples = Point[]
    for _ in 1:num_samples
        push!(samples, Point(rand(distr)))
    end

    return samples
end
function sample(event::CableCluster, model::CablesModel) end

function rescale(
    train_log_p_hist,
    test_log_p_hist,
    masked_proportion,
    model::NeymanScottModel
) return train_log_p_hist, test_log_p_hist end

function rescale(
    train_log_p_hist,
    test_log_p_hist,
    masked_proportion,
    model::PPSeq
)
    train_log_p_hist ./= ((1 - masked_proportion) * model.max_time * num_neurons(model))
    test_log_p_hist ./= (masked_proportion * model.max_time * num_neurons(model))

    return train_log_p_hist, test_log_p_hist
end


"""
Run Gibbs sampler on masked spike train. Alternate 
between imputing data in masked regions and updating
the model through classic gibbs_sample!(...) function.
"""
function masked_gibbs!(
    model::NeymanScottModel,
    masked_data::Vector{<: AbstractDatapoint},
    unmasked_data::Vector{<: AbstractDatapoint},
    masks::Vector{<: AbstractMask},
    initial_assignments::Vector{Int64},
    num_spike_resamples::Int64,
    samples_per_resample::Int64,
    extra_split_merge_moves::Int64,
    split_merge_window::Float64,
    save_every::Int64;
    verbose::Bool=true
)
    @assert length(unmasked_data) === length(initial_assignments)

    T = typeof(masked_data).parameters[1]

    sampled_data = T[]
    sampled_assignments = Int64[]

    # Compute proportion of the data that is masked.
    masked_prop = masked_proportion(model, masks)
    @show masked_prop

    # Create inverted masks to compute train log likelihood.
    #inv_masks = compute_complementary_masks(masks, num_neurons(model), model.max_time)
    inv_masks = compute_complementary_masks(masks, model)

    # Sanity check.
    assert_data_in_mask(masked_data, masks)
    assert_data_in_mask(unmasked_data, inv_masks)
    assert_data_not_in_mask(masked_data, inv_masks)
    assert_data_not_in_mask(unmasked_data, masks)

    # Compute log-likelihood of a homogeneous Poisson process in
    # the train and test sets.
    train_baseline = homogeneous_baseline_log_like(unmasked_data, inv_masks)
    test_baseline = homogeneous_baseline_log_like(masked_data, masks)

    @show train_baseline test_baseline

    num_unmasked = length(unmasked_data)
    assignment_hist = zeros(Int64, num_unmasked, 0)
    train_log_p_hist = Float64[]
    test_log_p_hist = Float64[]
    latent_event_hist = Vector[]
    globals_hist = []

    unmasked_assignments = initial_assignments

    for i = 1:num_spike_resamples

        # Sample new spikes in each masked region.
        sample_masked_data!(
            sampled_data,
            sampled_assignments,
            model,
            masks
        )

        # assert_spikes_in_mask(sampled_spikes, masks)
        # assert_spikes_not_in_mask(sampled_spikes, inv_masks)

        # Run gibbs sampler. Note that sufficient statistics are
        # recomputed at the beginning of gibb_sample!(...) so all
        # events will have the appropriate spike assignments / initialization
        (
            assignments,
            _assgn_hist,
            _lp_hist,
            _latents,
            _globals
        ) = 
        gibbs_sample!(
            model,
            vcat(unmasked_data, sampled_data),
            vcat(unmasked_assignments, sampled_assignments),
            samples_per_resample,
            extra_split_merge_moves,
            split_merge_window,
            save_every;
            verbose=false
        )

        # Update initial assignments for next Gibbs sampler run.
        unmasked_assignments .= view(assignments, 1:num_unmasked)

        # Save history
        assignment_hist = cat(
            assignment_hist,
            view(_assgn_hist, 1:num_unmasked, :),
            dims=2
        )

        # Evaluate model likelihood on observed spikes.        
        push!(
            train_log_p_hist,
            log_like(model, unmasked_data, inv_masks) - train_baseline
        )

        # Evaluate model likelihood on heldout spikes.
        push!(
            test_log_p_hist,
            log_like(model, masked_data, masks) - test_baseline
        )

        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

        verbose && print(i * samples_per_resample, "-")

    end

    verbose && println("Done")

    # Before returning, remove assignments assigned to imputed spikes.
    recompute!(model, unmasked_data, unmasked_assignments)

    # Rescale train and test log likelihoods.
    train_log_p_hist, test_log_p_hist = rescale(
        train_log_p_hist,
        test_log_p_hist,
        masked_prop,
        model
    )
    #train_log_p_hist ./= ((1 - masked_proportion) * model.max_time * num_neurons(model))
    #test_log_p_hist ./= (masked_proportion * model.max_time * num_neurons(model))

    return (
        unmasked_assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    )

end


"""
Annealed Gibbs sampling with masked data.
"""
function annealed_masked_gibbs!(
        model::NeymanScottModel,
        spikes::Vector{<: AbstractDatapoint},
        masks::Vector{<: AbstractMask},
        initial_assignments::Vector{Int64},
        num_anneals::Int64,
        max_temperature::Float64,
        num_spike_resamples_per_anneal::Int64,
        samples_per_resample::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=true
    )

    masked_spikes, unmasked_spikes = split_data_by_mask(spikes, masks)

    target_mean = mean(event_amplitude(priors(model)))
    target_var = var(event_amplitude(priors(model)))

    temperatures = exp10.(range(log10(max_temperature), 0, length=num_anneals))

    unmasked_assignments = fill(-1, length(unmasked_spikes))
    assignment_hist = zeros(Int64, length(unmasked_spikes), 0)
    train_log_p_hist = Float64[]
    test_log_p_hist = Float64[]
    latent_event_hist = []
    globals_hist = []

    for temp in temperatures
        
        # Print progress.
        verbose && println("TEMP:  ", temp)

        # Anneal prior on sequence amplitude.
        set_event_amplitude!(
            model, 
            specify_gamma(target_mean, target_var * temp)
        )

        # Recompute probability of introducing a new cluster.
        _gibbs_reset_model_probs(model)

        # Draw gibbs samples.
        (
            unmasked_assignments,
            _assgn,
            _train_hist,
            _test_hist,
            _latents,
            _globals
        ) = masked_gibbs!(
            model,
            masked_spikes,
            unmasked_spikes,
            masks,
            unmasked_assignments,
            num_spike_resamples_per_anneal,
            samples_per_resample,
            extra_split_merge_moves,
            split_merge_window,
            save_every;
            verbose=verbose
        )

        # Save samples.
        assignment_hist = cat(assignment_hist, _assgn, dims=2)
        append!(train_log_p_hist, _train_hist)
        append!(test_log_p_hist, _test_hist)
        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

    end

    return (
        unmasked_assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    )

end


function masked_gibbs!(
        model::PPSeq,
        spikes::Vector{Spike},
        masks::Vector{<: AbstractMask},
        initial_assignments::Vector{Int64},
        num_spike_resamples::Int64,
        samples_per_resample::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=true
    )

    masked_spikes, unmasked_spikes = split_data_by_mask(spikes, masks)

    return masked_gibbs!(
        model,
        masked_spikes,
        unmasked_spikes,
        masks,
        initial_assignments,
        num_spike_resamples,
        samples_per_resample,
        extra_split_merge_moves,
        split_merge_window,
        save_every;
        verbose=verbose
    )
end


"""
Impute missing data
"""
function sample_masked_data!(
    data::Vector{<: AbstractDatapoint},
    assignments::Vector{Int64},
    model::NeymanScottModel,
    masks::Vector{<: AbstractMask}
)
    empty!(data)
    empty!(assignments)

    # Sample background data
    for x in sample_background(model)
        for mask in masks
            if x in mask
                push!(data, x)
                push!(assignments, -1)
                break
            end
        end
    end

    # Sample event data
    for (k, event) in enumerate(events(model))
        # Assignment id for latent event.
        z = events(model).indices[k]

        for x in sample(event, model)
            # Check if spike is inside a masked region.
            for mask in masks
                if x in mask
                    push!(data, x)
                    push!(assignments, z)
                    break
                end
            end
        end
    end

    return data, assignments
end

"""
Impute missing data.
"""
function sample_masked_data(
    model::NeymanScottModel{N, D, E, P, G}, 
    masks::Vector{<: AbstractMask}
) where {N, D, E, P, G}
    data = D[]
    assgn = Int[]

    return sample_masked_data!(data, assgn, model, masks)
end



# ===========
#
# Helper functions to create masks, split spikes.
#
# ===========
function compute_complementary_masks(masks::Vector{CablesMask}, model::CablesModel)
    notimplemented()
end


function compute_complementary_masks(masks::Vector{GaussianMask}, model::GaussianNeymanScottModel)
    return [GaussianComplementMask(masks, bounds(model))]
end


function compute_complementary_masks(
    masks::Vector{SeqMask},
    model::PPSeq
)
    max_time = model.max_time
    num_neurons = num_neurons(model)

    inverted_masks = [[SeqMask(n, (0.0, max_time))] for n in 1:num_neurons]

    for mask in masks
        n = mask.neuron

        for i = 1:length(inverted_masks[n])
            inv_mask = inverted_masks[n][i]
            @assert inv_mask.neuron == n
            if (start(mask) >= start(inv_mask)) && (stop(mask) <= stop(inv_mask))
                deleteat!(inverted_masks[n], i)
                push!(
                    inverted_masks[n],
                    SeqMask(n, (start(inv_mask), start(mask)))
                )
                push!(
                    inverted_masks[n],
                    SeqMask(n, (stop(mask), stop(inv_mask)))
                )
                break
            end
            @assert (i + 1) != length(inverted_masks)
        end
    end

    return vcat(inverted_masks...)
end

function create_random_mask(
    model::PPSeq,
    mask_lengths::Real,
    percent_masked::Real
)
    num_neurons = num_neurons(model)
    max_time = volume(model)

    @assert num_neurons > 0
    @assert max_time > mask_lengths
    @assert mask_lengths > 0
    @assert 0 <= percent_masked < 100

    T_masked = percent_masked * max_time * num_neurons / 100.0
    n_masks = Int(round(T_masked / mask_lengths))

    intervals = Tuple{Float64, Float64}[]
    for start in range(0, max_time - mask_lengths, step=mask_lengths)
        push!(intervals, (start, start + mask_lengths))
    end

    masks = [
        SeqMask(n, (t1, t2))
        for (n, interval) in Iterators.product(1:num_neurons, intervals)
    ]
    
    return sample(masks, n_masks, replace=false)
end
function create_random_mask(
    model::GaussianNeymanScottModel,
    mask_radius::Real,
    percent_masked::Real
)
    xlim, ylim = bounds(model)
    area = (xlim - mask_radius) * (ylim - mask_radius)  # Don't include boundary

    # Fill box with disjoint masks
    masks = GaussianMask[]
    x = mask_radius
    while x <= (xlim - mask_radius)
        y = mask_radius
        while y <= (ylim - mask_radius)
            push!(masks, GaussianMask((x, y), mask_radius))
            y += 2*mask_radius
        end
        x += 2*mask_radius
    end

    # Sample masks
    num_masks = floor(Int, area * percent_masked / (π*mask_radius^2))
    return sample(masks, num_masks, replace=false)
end

function create_blocked_mask(model::PPSeq)

    num_neurons = num_neurons(model)
    max_time = volume(model)

    masked_neurons = sample(1:num_neurons, num_neurons ÷ 2, replace=false)

    masked_intervals = vcat(
        repeat([(0.0, max_time / 2)], div(num_neurons ÷ 2, 2)),
        repeat([(max_time / 2, max_time)], cld(num_neurons ÷ 2, 2))
    )

    return [
        SeqMask(n, interval) for
        (n, interval) in zip(masked_neurons, masked_intervals)
    ]
end


function split_data_by_mask(
    data::Vector{<: AbstractDatapoint},
    masks::Vector{<: AbstractMask}
)
    # Save list of spikes that are masked out.
    masked_data = []
    unmasked_data = []

    for x in data
        # See if x falls within any masked region.
        if x in masks
            push!(masked_data, x)
        else  # Mark x as unmasked if no match was found.
            push!(unmasked_data, x)
        end
    end

    return masked_data, unmasked_data
end


function assert_data_in_mask(
    data::Vector{<: AbstractDatapoint}, 
    masks::Vector{<: AbstractMask}
)
    for x in data
        if !(x in masks)
            @show x
            @assert false "datapoint is falsely excluded from mask..."
        end
    end
end


function assert_data_not_in_mask(
    data::Vector{<: AbstractDatapoint},
    masks::Vector{<: AbstractMask}
)
    # Check that all spikes are in masked region.
    for x in data
        if x in masks
            @show x
            @assert false "datapoint is falsely included in mask..."
        end
    end
end


function in(x::AbstractDatapoint, masks::Vector{<: AbstractMask})
    for msk in masks
        if x in msk
            return true
        end
    end
    return false  # Not in any of the masks
end
