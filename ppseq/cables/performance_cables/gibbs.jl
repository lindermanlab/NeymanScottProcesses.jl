import Random
using PointProcessSequences


"""
Generates data with
    event_rate * event_amplitude_mean + bkgd_amplitude_mean
cables per second. By default, generates 4.5 cables per second.
"""
function run_gibbs(
    ; max_time=100.0,
    seed=123,
    event_rate=0.1,
    event_amplitude_mean=30.0,
    bkgd_amplitude_mean=1.5,
    nthreads=1,
)
    Random.seed!(seed)

    # Specify parameters
    max_radius = Inf
    num_nodes = 10
    num_marks = 100

    event_rate = event_rate
    event_amplitude_mean = event_amplitude_mean
    event_amplitude_var = 1e-1
    bkgd_amplitude_mean = bkgd_amplitude_mean
    bkgd_amplitude_var = 1e-15

    node_concentration = 5e-2 * ones(num_nodes)
    mark_concentration = 1e-2 * ones(num_marks)

    bkgd_node_concentration = 1.0 * ones(num_nodes)
    bkgd_mark_concentration = 1e1 * ones(num_marks, num_nodes)

    variance_pseudo_obs = 100.0
    variance_scale = 10.0 * variance_pseudo_obs

    priors = CablesPriors(
        event_rate,
        specify_gamma(event_amplitude_mean, event_amplitude_var),
        specify_gamma(bkgd_amplitude_mean, bkgd_amplitude_var),

        node_concentration,
        mark_concentration,
        bkgd_node_concentration,
        bkgd_mark_concentration,

        variance_scale,
        variance_pseudo_obs,
    )

    # Generate model and sample data
    gen_model = CablesModel(
        max_time,
        max_radius,
        priors
    )
    data, assignments, events = sample(gen_model, resample_latents=true)
    #@show length(data)

    # Create a new model to fit this data
    model = CablesModel(max_time, max_radius, priors)
    distr_model = make_distributed(model, nthreads)
    runtime = @elapsed (
        est_assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) = gibbs_sample!(
        distr_model,
        data,
        fill(-1, length(data)),
        verbose=true,
        num_samples=100,
        save_every=10,
    )

    return runtime, length(data), length(events)
end