
# colors
nice_colors = Dict(
    :orange => "#FF4400",
    :purple => "#800080",
    :gray => "#A4A4A4",
    :dark_gray => "#1A1A1A",
    :saffron => "#FEB209",
    :shamrock => "#02c14d",
)


# ===========
#
#  RASTERS
#
# ===========

function plot_init_raster(
        dataset_id::String,
        sweep_id::Int64,
        job_id::Int64,
    )
    spikes, _, _ = load_dataset(dataset_id)
    results = load_results(dataset_id, sweep_id, job_id)
    init_model = load_init_model(dataset_id, sweep_id, job_id)
    events = PointProcessSequences.event_list_summary(init_model.sequence_events)
    neuron_order = sortperm_neurons(init_model.globals)
    plot_raster(spikes, events, results[:initial_assignments], neuron_order)
end


function plot_raster(
        dataset_id::String,
        sweep_id::Int64,
        job_id::Int64,
        sample_idx::Int64;
        kwargs...
    )

    spikes, _, _ = load_dataset(dataset_id)
    results = load_results(dataset_id, sweep_id, job_id)
    masks = load_masks(dataset_id, sweep_id, job_id)
    
    assignment_hist = results[:assignment_hist]

    if sample_idx < 0
        sample_idx = size(assignment_hist, 2) 
    end

    globals = results[:globals_hist][sample_idx]
    events = results[:latent_event_hist][sample_idx]
    neuron_order = sortperm_neurons(globals)

    if !isempty(masks)
        (masked_spikes, unmasked_spikes) = 
            split_spikes_by_mask(spikes, masks)
        unmasked_assignments = assignment_hist[1:length(unmasked_spikes), sample_idx]
        fig = plot_raster(
            unmasked_spikes, events, unmasked_assignments, neuron_order; kwargs...)
        plot_raster(
            fig, masked_spikes; c="g")
        plot_masks(fig, masks)

    else
        fig = plot_raster(
            spikes,
            events,
            assignment_hist[:, sample_idx],
            neuron_order;
            kwargs...
        )
    end

    plt.title(@sprintf("%s, sweep %i, job %05i", dataset_id, sweep_id, job_id))

    return fig
end

plot_raster(spikes::Vector{Spike}; kwargs...) = (
    plot_raster(plt.figure(), spikes; kwargs...)
)

function plot_raster(fig::PyPlot.Figure, spikes::Vector{Spike}; kwargs...)
    _x, _y = zeros(length(spikes)), zeros(length(spikes))
    for (i, s) in enumerate(spikes)
        _x[i] = s.timestamp
        _y[i] = s.neuron
    end
    (length(fig.axes) < 1) && fig.add_subplot(1, 1, 1)
    fig.axes[1].scatter(_x, _y; s=4, kwargs...)
    return fig
end


function plot_raster(
        spikes::Vector{Spike},
        events::Vector{EventSummaryInfo},
        assignments::Vector{Int64},
        neuron_order::Vector{Int64};
        kwargs...
    )
    plot_raster(
        plt.figure(),
        spikes,
        events,
        assignments,
        neuron_order;
        kwargs...
    )
end


function plot_raster(
        fig::PyPlot.Figure,
        spikes::Vector{Spike},
        events::Vector{EventSummaryInfo},
        assignments::Vector{Int64},
        neuron_order::Vector{Int64};
        kwargs...
    )
    
    _x, _y = zeros(length(spikes)), zeros(length(spikes))
    _c = String[]

    typemap = Dict((e.assignment_id => e.seq_type) for e in events)

    for (i, s) in enumerate(spikes)
        _x[i] = s.timestamp
        _y[i] = neuron_order[s.neuron]

        if assignments[i] == -1
            push!(_c, "k")
        elseif typemap[assignments[i]] == 1
            push!(_c, "b")
        elseif typemap[assignments[i]] == 2
            push!(_c, "r")
        elseif typemap[assignments[i]] == 3
            push!(_c, "g")
        end
    end

    (length(fig.axes) < 1) && fig.add_subplot(1, 1, 1)
    fig.axes[1].scatter(_x, _y; c=_c, s=4, kwargs...)
    return fig
end

plot_masks(masks::Vector{Mask}; kwargs...) = plot_masks(plt.subplots(1, 1)[1], masks; kwargs...)

function plot_masks(fig::PyPlot.Figure, masks::Vector{Mask}; color="k")
    for (n, (t0, t1)) in masks
        fig.axes[1].plot([t0, t1], [n, n], color=color, lw=3, alpha=.4)
    end
    return fig
end

# ===========
#
#  EVENTS
#
# ===========

plot_events(dataset_id::String, sweep_id::Int64, job_id::Int64) = (
    plot_events(load_model(dataset_id, sweep_id, job_id))
)

function plot_events(model::PPSeq)

    fig = plt.figure()

    colors = ["b", "r", "g"]

    for (k, event) in enumerate(model.sequence_events)
        τ = event.sampled_timestamp
        A = event.sampled_amplitude
        r = event.sampled_type
        plt.plot([τ, τ], [0., A], color=colors[r])
    end

    return fig
end


function plot_num_events_history(dataset_id::String, sweep_id::Int64, job_id::Int64)

    results = load_results(dataset_id, sweep_id, job_id)
    config = load_config(dataset_id, sweep_id, job_id)

    fig1 = plt.figure()
    y = [length(ev) for ev in results[:anneal_latent_event_hist]]
    total_samples = config[:samples_per_anneal] * config[:num_anneals]
    x = range(
        1, total_samples, length=length(y)
    )
    plt.plot(x, y)
    for x in range(1, total_samples, step=config[:samples_per_anneal])
        fig1.axes[1].axvline(x; dashes=(2, 2), color="k", alpha=.5)
    end
    plt.ylabel("num events")
    plt.xlabel("samples")
    plt.title("num events during anneal")

    fig2 = plt.figure()
    R = config[:num_sequence_types]
    ys = zeros(length(results[:anneal_latent_event_hist]), R)

    for (k, ev) in enumerate(results[:anneal_latent_event_hist])
        for event in ev
            ys[k, event.seq_type] += 1
        end
        # for c in results[:anneal_assignment_hist][:, k]
        #     (c == -1) && continue
        #     for event in ev
        #         if event.assignment_id == c
        #             ys[k, event.seq_type] += 1
        #         end
        #     end
        # end
    end
    plt.plot(x, ys)

    fig3 = plt.figure()
    plt.plot([length(ev) for ev in results[:latent_event_hist]])
    plt.ylabel("num events")
    plt.xlabel("samples")
    plt.title("num events after anneal")

    return fig1, fig2, fig3
end

function plot_num_events_histogram(
        fig::PyPlot.Figure,
        dataset_id::String,
        sweep_id::Int64,
        job_id::Int64;
        kwargs...
    )
    latent_event_hist = load_results(dataset_id, sweep_id, job_id)[:latent_event_hist]
    num_events = [length(ev) for ev in latent_event_hist]
    burnin = length(num_events) ÷ 2
    fig.axes[1].hist(num_events[burnin:end]; kwargs...)
    return fig
end

function plot_num_events_histogram(dataset_id::String, sweep_id::Int64, job_id::Int64; kwargs...)
    fig, _ = plt.subplots(1, 1)
    plot_num_events_histogram(fig, dataset_id, sweep_id, job_id; kwargs...)
end


# ===========
#
#  LIKELIHOOD
#
# ===========

function plot_log_likes(dataset_id::String, sweep_id::Int64, job_id::Int64)
    
    results = load_results(dataset_id, sweep_id, job_id)
    fig = plt.figure()
    if (:masks in keys(results)) && !isempty(results[:masks])
        train_ll = results[:train_log_p_hist]
        test_ll = results[:test_log_p_hist]
        train_x = range(1, length(train_ll), length=length(train_ll)) 
        test_x = range(1, length(train_ll), length=length(test_ll))
        plt.plot(train_x, train_ll, "-b", label="train set")
        plt.plot(test_x, test_ll, "-r", label="test set")
    else
        plt.plot(results[:log_p_hist], "-b", label="train set")
    end
    plt.legend()
    plt.title(@sprintf("%s, sweep %i, job %05i", dataset_id, sweep_id, job_id))
    return fig
end


# ===========
#
#  SPIKE CO-OCCUPANCY
#
# ===========

function plot_spike_co_occupancy(
        dataset_id::String,
        sweep_id::Int64,
        job_id::Int64;
        kwargs...
    )

    P, Psort = spike_co_occupancy(
        load_results(dataset_id, sweep_id, job_id)[:assignment_hist]
    )

    fig1 = plt.figure()
    plt.imshow(P, kwargs...)
    plt.title(@sprintf("UNSORTED || %s, sweep %i, job %05i", dataset_id, sweep_id, job_id))

    fig2 = plt.figure()
    plt.imshow(Psort, kwargs...)
    plt.title(@sprintf("SORTED || %s, sweep %i, job %05i", dataset_id, sweep_id, job_id))

    return fig1, fig2
end


# ===========
#
#  CONFIDENCE INTERVALS
#
# ===========

function plot_annealing_schedule(
        dataset_id::String,
        sweep_id::Int64,
        job_id::Int64,
    )
    
    config = load_config(dataset_id, sweep_id, job_id)
    temperatures = Float64[]
    new_cluster_log_probs = Float64[]
    bkgd_log_probs = Float64[]
    prob_new_cluster = Float64[]

    log_max_temp = log10(config[:max_temperature])
    mean_amp = config[:mean_event_amplitude]
    var_amp = config[:var_event_amplitude]
    seq_rate = config[:seq_event_rate]
    bkgd_rate = config[:mean_bkgd_spike_rate]

    for temp in exp10.(range(0.0, log_max_temp, length=config[:num_anneals]))

        g = specify_gamma(mean_amp, var_amp * temp)

        push!(
            new_cluster_log_probs,
            log(seq_rate) + log(g.α) + g.α * (log(g.β) - log(1 + g.β))
        )
        push!(
            bkgd_log_probs,
            log(bkgd_rate) + log(1 + g.β)
        )
        push!(temperatures, temp)

        push!(
            prob_new_cluster,
            softmax([
                new_cluster_log_probs[end],
                bkgd_log_probs[end]
            ])[1]
        )

        @assert abs(mean_amp - g.α / g.β) < 1e-6
        @assert abs(var_amp * temp - g.α / g.β^2) < 1e-6

    end

    plt.figure()
    plt.plot(temperatures, prob_new_cluster, ".-")
    plt.ylabel("probability of new cluster")
    plt.xlabel("temperature")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e-8, 1.0])
    plt.legend()

end
