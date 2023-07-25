
# ==================================================== #
# ===                                              === #
# === Methods to create, resample sequence events. === #
# ===                                              === #
# ==================================================== #

constructor_args(e::SeqEvent) = size(e.summed_potentials)


"""Constructs empty sequence event."""
function SeqEvent(
        num_sequence_types::Int64,
        num_warp_values::Int64
    )

    return SeqEvent(
        0,
        zeros(num_sequence_types, num_warp_values),  # summed potentials
        zeros(num_sequence_types, num_warp_values),  # summed precisions
        zeros(num_sequence_types, num_warp_values),  # summed log Z
        zeros(num_sequence_types, num_warp_values),  # posterior on sequence type.
        -1,   # sampled_type is ignored until spike_count > 0
        -1,   # sampled_warp is ignored until spike_count > 0
        0.0,  # sampled_timestamp is ignored until spike_count > 0
        0.0   # sampled_amplitude is ignored until spike_count > 0
    )
end


"""Resets event to be empty."""
function reset!(e::SeqEvent)
    e.spike_count = 0
    fill!(e.summed_potentials, 0)
    fill!(e.summed_precisions, 0)
    fill!(e.summed_logZ, 0)
    fill!(e.seq_type_posterior, 0)
end


"""
Remove spike `s` from event at index `k`.
"""
function remove_datapoint!(
    model::PPSeq,
    s::Spike,
    k::Int64
)
    
    # Nothing to do if spike is already in background.
    (k == -1) && return

    # Remove the contributions of datapoint i to the
    # sufficient statistics of sequence event k.
    t = s.timestamp
    n = s.neuron
    event = model.sequence_events[k]

    log_p_neuron = model.globals.neuron_response_log_proportions
    offsets = model.globals.neuron_response_offsets
    widths = model.globals.neuron_response_widths
    warps = model.priors.warp_values

    # If this is the last spike in the event, we can return early.
    (event.spike_count == 1) && (return remove_event!(model.sequence_events, k))

    # Otherwise, subtract off the sufficient statistics.
    event.spike_count -= 1
    
    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            # Compute 1D precision and potential.
            v = 1 / (widths[n, r] * warps[w]^2)
            m = (t - offsets[n, r] * warps[w]) * v
            sv = event.summed_precisions[r, w]

            # Subtract sufficient statistics.
            event.summed_potentials[r, w] -= m
            event.summed_precisions[r, w] = max(0, sv - v) # prevent negative precision.
            event.summed_logZ[r, w] -= (log_p_neuron[n, r] - gauss_info_logZ(m, v))            
            event.seq_type_posterior[r, w] = (
                model.globals.seq_type_log_proportions[r]
                + model.priors.warp_log_proportions[w]
                + event.summed_logZ[r, w]
                + gauss_info_logZ(
                    event.summed_potentials[r, w],
                    event.summed_precisions[r, w])
            )
        end
    end

    event.seq_type_posterior .-= logsumexp(event.seq_type_posterior)

end


"""
Add spike `s` to event at index `k`. Return assignment index `k`.
"""
function add_datapoint!(
        model::PPSeq,
        s::Spike,
        k::Int64;
        recompute_posterior::Bool=true
    )

    t = s.timestamp
    n = s.neuron
    event = model.sequence_events[k]

    log_p_neuron = model.globals.neuron_response_log_proportions
    offsets = model.globals.neuron_response_offsets
    widths = model.globals.neuron_response_widths
    warps = model.priors.warp_values

    event.spike_count += 1
    
    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            # Compute 1D precision and potential.
            v = 1 / (widths[n, r] * warps[w]^2)
            m = (t - offsets[n, r] * warps[w]) * v
            
            # Add sufficient statistics.
            event.summed_potentials[r, w] += m
            event.summed_precisions[r, w] += v
            event.summed_logZ[r, w] += (log_p_neuron[n, r] - gauss_info_logZ(m, v))
        end
    end

    if recompute_posterior
        set_posterior!(model, k)
    end

    return k
end


function set_posterior!(
    model::PPSeq,
    k::Int64;
)
    event = model.sequence_events[k]

    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            event.seq_type_posterior[r, w] = (
                model.globals.seq_type_log_proportions[r]
                + model.priors.warp_log_proportions[w]
                + event.summed_logZ[r, w]
                + gauss_info_logZ(
                    event.summed_potentials[r, w],
                    event.summed_precisions[r, w])
            )
        end
    end

    event.seq_type_posterior .-= logsumexp(event.seq_type_posterior)
end


"""
Returns a vector of EventSummaryInfo structs
summarizing latent events and throwing away
sufficient statistics.
"""
function event_list_summary(model::PPSeq)
    infos = EventSummaryInfo[]
    ev = model.sequence_events
    warp_vals = model.priors.warp_values

    for ind in ev.indices
        push!(
            infos,
            EventSummaryInfo(
                ind,
                ev[ind].sampled_timestamp,
                ev[ind].sampled_type,
                warp_vals[ev[ind].sampled_warp],
                ev[ind].sampled_amplitude
            )
        )
    end

    return infos
end