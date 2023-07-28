function gibbs_sample_event!(event::SeqEvent, model::PPSeq)
    # Grab length-R vector (already pre-allocated).
    log_probs = model.buffers[:RW]

    # We should only be resampling non-empty events.
    @assert (event.spike_count > 0)

    # Sample sequence type.
    log_probs .= event.seq_type_posterior
    ind = sample_logprobs!(vec(log_probs))
    r, w = Tuple(CartesianIndices(size(log_probs))[ind])
    event.sampled_type = r
    event.sampled_warp = w

    # Sample event time, t ~ N(μ, σ2), given sequence type.
    σ2 = 1 / event.summed_precisions[r, w]
    μ = event.summed_potentials[r, w] * σ2
    event.sampled_timestamp = μ + randn() * sqrt(σ2)

    # Sample event amplitude.
    event.sampled_amplitude = rand(
        posterior(
            event.spike_count,
            model.priors.seq_event_amplitude
        )
    ) # Poisson - gamma conjugate pair.
end


function gibbs_update_globals!(
    model::PPSeq,
    spikes::Vector{Spike},
    assignments::AbstractVector{Int64}
)

    K = num_events(model)
    N = num_neurons(model)
    R = num_sequence_types(model)

    priors = model.priors
    globals = model.globals

    # === RESAMPLE BACKGROUND SPIKE PARAMETERS === #

    num_bkgd_spikes = 0
    bkgd_counts = zeros(N)

    for (i, x) in enumerate(spikes)
        if assignments[i] < 0
            num_bkgd_spikes += 1
            bkgd_counts[x.neuron] += 1
        end
    end

    # Sample proportions of background spikes across neurons
    # as a probability vector then map to log space for
    # future computations.
    rand!(
        posterior(
            bkgd_counts, priors.bkgd_proportions
        ),
        globals.bkgd_log_proportions
    ) # multinomial - symmetric dirichlet conjugate pair.
    map!(
        log,
        globals.bkgd_log_proportions,
        globals.bkgd_log_proportions
    )

    # Sample the rate of background spikes. Here, we need to adjust
    # for the length of the time interval, max_time. The scaling
    # property of the gamma distribution implies that:
    #
    #   bkgd_amp * max_time ~ RateGamma(α_bkgd, β_bkgd / max_time)
    #
    # If we observe num_bkgd, then the posterior (by Poisson-gamma
    # conjugacy) is:
    #
    #   bkgd_amp * T | num_bkgd_spikes ~ RateGamma(num_bkgd + α_bkgd, 1 + β_bkgd / T)
    #
    # Now apply the gamma scaling property again, dividing by T this
    # time, so we get:
    #
    #   bkgd_amp | num_bkgd_spikes ~ RateGamma(num_bkgd + α_bkgd, T + β_bkgd)
    #
    globals.bkgd_amplitude = rand(
        RateGamma(
            num_bkgd_spikes + priors.bkgd_amplitude.α,
            priors.bkgd_amplitude.β + model.max_time
        )
    )


    # === RESAMPLE SEQUENCE TYPE PROPORTIONS === #

    seq_type_counts = zeros(R)
    for event in model.sequence_events
        seq_type_counts[event.sampled_type] += 1
    end

    rand!(
        posterior(
            seq_type_counts, priors.seq_type_proportions
        ),
        model.globals.seq_type_log_proportions
    )  # multinomial - symmetric dirichlet conjugate pair.
    map!(
        log,
        globals.seq_type_log_proportions,
        globals.seq_type_log_proportions
    )

    # === RESAMPLE NEURON RESPONSE PROFILES === #

    # TODO -- preallocate arrays for this? Or is it not worth it?
    spk_count = zeros(Int64, N, R)
    spk_fm = zeros(N, R)
    spk_sm = zeros(N, R)

    for (i, x) = enumerate(spikes)

        # spike assignment to latent event.
        k = assignments[i]
        
        # skip if spike is assigned to background.
        (k < 0) && continue

        event = model.sequence_events[k]
        n = x.neuron
        r = event.sampled_type
        w = event.sampled_warp
        offset = (x.timestamp - event.sampled_timestamp) / model.priors.warp_values[w]

        # compute sufficient statistics
        spk_count[n, r] += 1
        spk_fm[n, r] += offset
        spk_sm[n, r] += offset * offset

    end

    for r = 1:R

        rand!(
            posterior(
                view(spk_count, :, r),
                priors.neuron_response_proportions
            ),
            view(globals.neuron_response_log_proportions, :, r)
        ) # multinomial - dirichlet conjugate pair.

        for n = 1:N

            (
                globals.neuron_response_offsets[n, r],
                globals.neuron_response_widths[n, r]
            ) =
            rand(
                posterior(
                    spk_count[n, r],
                    spk_fm[n, r],
                    spk_sm[n, r],
                    priors.neuron_response_profile
                )
            ) # normal - norm-inv-chi-squared conjugate pair.

        end

    end

    map!(
        log,
        globals.neuron_response_log_proportions,
        globals.neuron_response_log_proportions
    )

end