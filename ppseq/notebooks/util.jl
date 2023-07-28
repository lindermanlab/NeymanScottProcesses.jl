using Random
using PointProcessSequences
using StatsBase

times(spikes) = [s.timestamp for s in spikes]
neurons(spikes) = [s.neuron for s in spikes]

function generate_data(;
    seed=nothing, 
    max_time=100, 
    num_neurons=100,
    
    bkgd_rate=0.01,  # per second
        
    event_rate=0.1,  # per second
    amplitude=10,
    
    seq_duration=1,  # offsets go from -dur/2 to dur/2
    seq_width=0.2,  # standard deviation of firing time
)
    (!isnothing(seed)) && Random.seed!(seed)

    events = []
    spikes = Spike[]
    ids = []

    num_bkgd_spikes = 
        floor(Int, bkgd_rate * max_time)
    
    num_seq = floor(Int, event_rate * max_time)
    offsets = range(
        -seq_duration/2, seq_duration/2, 
        length=num_neurons
    )

    # Add background spikes
    for _ = 1:num_bkgd_spikes
        spk = Spike(rand(1:num_neurons), rand()*max_time)
        push!(spikes, spk)
        push!(ids, -1)
    end

    # Create sequences
    for _ = 1:num_seq
        event_time = rand()*max_time
        push!(events, event_time)
    end

    # Add spikes to sequences
    for _ = 1:num_seq*amplitude
        # Sample event
        event = rand(1:num_seq)

        # Sample neuron
        neuron = rand(1:num_neurons)

        # Sample time
        time = events[event] + 
               offsets[neuron] + 
               randn() * seq_width
        
        time = max(0, min(max_time, time))

        push!(spikes, Spike(neuron, time))
        push!(ids, event)
    end
    
    return events, spikes, ids
end


function plot_raster(spikes, labels; model=nothing, events=nothing)
    @assert isnothing(model) || isnothing(events)

    map_colors(id) = (id == -1) ? "black" : id
    p = plot(
        times(spikes), neurons(spikes), seriestype=:scatter,
        legend=false, grid=false, color=map_colors.(labels), size=(800, 200),
        ms=2, markerstrokewidth=0
    )
    
    if !isnothing(model)
        events = [
            (ind, model.sequence_events[ind].sampled_timestamp) 
            for ind in model.sequence_events.indices
        ]
    end
    if !isnothing(events)
        plot!(
            p, [e[2] for e in events], fill(-10, length(events)), seriestype=:scatter,
            color=[e[1] for e in events], ms=5, marker=:utriangle
        )
    end
    
    return p
end

function eval(true_events, est_events; thresh=2, nsamples=50)
    if length(est_events) == 0
        return 0.0
    end

    loss = 0.0
    for _ = 1:nsamples
        if length(est_events) > length(true_events)
            filtered_est = sample(est_events, length(true_events), replace=false)
        else
            filtered_est = est_events
        end
        for e in true_events
            loss += (minimum(abs.(e .- filtered_est)) < thresh)
        end
    end
    return loss / length(true_events) / nsamples
end

function eval(true_events, model::PPSeq)
    est_events = [e.sampled_timestamp for e in model.sequence_events]
    return eval(true_events, est_events)
end


function init_model(
    max_time, 
    num_neurons,
    num_motifs,
    bkgd_rate,
    amplitude,
    seq_width,
    event_rate;
    max_sequence_length=10.0
)

    seq_type_proportions = SymmetricDirichlet(1.0, num_motifs)

    bkgd_proportions = SymmetricDirichlet(1.0, num_neurons)
    neuron_response_proportions = SymmetricDirichlet(0.1, num_neurons)

    bkgd_amplitude = specify_gamma(   
        bkgd_rate,     # mean of gamma; α / β
        bkgd_rate * 1e-6,     # variance of gamma; α / β²
    )

    seq_event_amplitude = specify_gamma(amplitude, amplitude^2)
    neuron_response_profile = NormalInvChisq(
        1.0,   # κ, pseudo-observations of prior mean
        0.0,   # m, prior mean for offset parameter
        2.0,   # ν, pseudo-observations of prior variance
        seq_width    # s2, prior variance
    )
    return PPSeq(
        max_time,
        max_sequence_length,
        # priors
        event_rate,
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )
end


function init_distr_model(
    nthreads,
    max_time, 
    num_neurons,
    num_motifs,
    bkgd_rate,
    amplitude,
    seq_width,
    event_rate;
    max_sequence_length=10.0
)

    seq_type_proportions = SymmetricDirichlet(1.0, num_motifs)

    bkgd_proportions = SymmetricDirichlet(1.0, num_neurons)
    neuron_response_proportions = SymmetricDirichlet(0.1, num_neurons)

    bkgd_amplitude = specify_gamma(   
        bkgd_rate,     # mean of gamma; α / β
        bkgd_rate * 1e-5,     # variance of gamma; α / β²
    )

    seq_event_amplitude = specify_gamma(amplitude, amplitude^2 / 2^2)
    neuron_response_profile = NormalInvChisq(
        1.0,   # κ, pseudo-observations of prior mean
        0.0,   # m, prior mean for offset parameter
        2.0,   # ν, pseudo-observations of prior variance
        seq_width    # s2, prior variance
    )

    return DistributedPPSeq(
        # constants
        nthreads,  # threads
        max_time,
        1.0,

        # warp parameters
        1,
        1.0,
        1.0,

        # priors
        event_rate,
        SymmetricDirichlet(1.0, 1),
        seq_event_amplitude,
        SymmetricDirichlet(1.0, num_neurons),
        neuron_response_profile,
        bkgd_amplitude,
        SymmetricDirichlet(1.0, num_neurons)
    )
end