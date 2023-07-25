function set_bkgd_amplitude!(model::AbstractModel, bkgd_amplitude::RateGamma)
    model.priors.bkgd_amplitude = bkgd_amplitude
    model.globals.bkgd_rate = bkgd_amplitude.α / bkgd_amplitude.β
end
function set_bkgd_amplitude!(model::DistributedNeymanScottModel, bkgd_amplitude::RateGamma)
    # We only need to do this for the primary model, since all models share
    # priors and globals.
    model.primary_model.priors = bkgd_amplitude
    model.globals.bkgd_rate = bkgd_amplitude.α / bkgd_amplitude.β
end
function set_bkgd_amplitude!(model::PPSeq, bkgd_amplitude::RateGamma)
    model.priors.bkgd_amplitude = bkgd_amplitude
    model.globals.bkgd_amplitude = bkgd_amplitude.α / bkgd_amplitude.β
end


function set_event_amplitude!(model::AbstractModel, event_amplitude::RateGamma)
    model.priors.event_amplitude = event_amplitude
end
function set_event_amplitude!(model::PPSeq, event_amplitude::RateGamma)
    model.priors.seq_event_amplitude = event_amplitude
end
function set_event_amplitude!(model::DistributedNeymanScottModel, event_amplitude::RateGamma)
    # We only need to do this for the primary model, since all models share
    # priors and globals.
    model.primary_model.priors.event_amplitude = event_amplitude
end

volume(model::GaussianNeymanScottModel) = area(model)
volume(model::PPSeq) = model.max_time


"""
Annealed Gibbs sampling, to remove small amplitude events.
"""
function annealed_gibbs!(
    model::AbstractModel,
    spikes::Vector{<:AbstractDatapoint},
    initial_assignments::Vector{Int64};
    num_anneals=3,
    samples_per_anneal=100,
    max_temperature=100.0,
    extra_split_merge_moves=0,
    split_merge_window=1.0,
    save_every=1,
    verbose=false,
    anneal_background=false,
    save_set=[:latents, :globals, :assignments],
)
    return annealed_gibbs!(
        model,
        spikes,
        initial_assignments,
        num_anneals,
        samples_per_anneal,
        max_temperature,
        extra_split_merge_moves,
        split_merge_window,
        save_every,
        verbose=verbose,
        anneal_background=anneal_background,
        save_set=save_set,
    )
end

function annealed_gibbs!(
        model::AbstractModel,
        spikes::Vector{<:AbstractDatapoint},
        initial_assignments::Vector{Int64},
        num_anneals::Int64,
        samples_per_anneal::Int64,
        max_temperature::Float64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=false,
        anneal_background::Bool=false,
        save_set=[:latents, :globals, :assignments],
    )

    # Initialize storage.
    assignment_hist = zeros(Int64, length(spikes), 0)
    log_p_hist = Float64[]
    latent_event_hist = Vector{Any}[]
    globals_hist = AbstractGlobals[]

    # Return early if no samples.
    if num_anneals == 0
        return (
            initial_assignments,
            assignment_hist,
            log_p_hist,
            latent_event_hist,
            globals_hist
        )
    end

    # Final amplitude for anneal.
    target_mean = mean(event_amplitude(priors(model)))
    target_var = var(event_amplitude(priors(model)))

    target_bkgd_mean = mean(bkgd_amplitude(priors(model)))
    target_bkgd_var = var(bkgd_amplitude(priors(model)))

    # Begin annealing.
    temperatures = exp10.(range(log10(max_temperature), 0, length=num_anneals))
    assignments = initial_assignments

    for temp in temperatures
        
        # Print progress.
        verbose && println("TEMP:  ", temp)

        if anneal_background
            # Anneal prior on background rate
            set_bkgd_amplitude!(
                model,
                specify_gamma(target_bkgd_mean / temp, target_bkgd_var)
            )
        else
            # Anneal prior on sequence amplitude
            set_event_amplitude!(
                model, 
                specify_gamma(target_mean, target_var * temp)
            )
        end

        # Recompute the new cluster and background probabilities
        _gibbs_reset_model_probs(model)

        # Draw gibbs samples.
        (
            assignments,
            _assgn,
            _logp,
            _latents,
            _globals
        ) = gibbs_sample!(
            model,
            spikes,
            assignments,
            samples_per_anneal,
            extra_split_merge_moves,
            split_merge_window,
            save_every;
            verbose=verbose,
            save_set=save_set,
        )

        # Save samples.
        assignment_hist = cat(assignment_hist, _assgn, dims=2)
        append!(log_p_hist, _logp)
        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

    end

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )

end
