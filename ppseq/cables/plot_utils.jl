import PyPlot: plt


"""Plot log-likelihood and, optionally, number of latent events over time."""
function plot_loglike(loglike_hist, latent_event_hist=nothing)
    should_plot_events = !(latent_event_hist === nothing)

    plt.figure()

    should_plot_events && plt.subplot(1, 2, 1)
    plt.plot(loglike_hist)
    plt.title("log likelihood")
    plt.xlabel("number of samples")

    if should_plot_events
        plt.subplot(1, 2, 2)
        plt.plot([length(latent_event_hist[i]) for i in 1:length(latent_event_hist)])
        plt.title("number of latent events")
        plt.xlabel("number of samples")
    end
end


"""Plot cables and their assignments."""
function plot_cables(cables, assgn; est_assgn=nothing)
    num_cables = length(cables)
    times = [c.position for c in cables]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(
        [times[assgn .== k] for k in sort(unique(assgn))], 
        bins=collect(0.0:maximum(times)/50:maximum(times)), 
        rwidth=0.5, stacked=true
    )

    if !(est_assgn === nothing)
        plt.subplot(2, 1, 2)
        plt.hist(
            [times[est_assgn .== k] for k in sort(unique(est_assgn))], 
            bins=collect(0.0:maximum(times)/50:maximum(times)), 
            rwidth=0.5, stacked=true
        )
    end
    #plt.scatter(ts.(cables), rand(num_cables), c=assgn .+ 2, s=6)
    #plt.ylim(-1, 10)
end

function plot_word_intensity(
    cables
)
end
