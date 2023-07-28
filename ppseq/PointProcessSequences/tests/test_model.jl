using PointProcessSequences
using Test: @test

include("./util.jl")


function test_gibbs_for_crashes()
    # generate data and model
    model, _ = make_test_spikes_and_model(max_time=10.0)
    spikes, _, _ = sample(model, resample_latents=true)

    # generate distributed model
    regular_model = deepcopy(model)

    # set parameters
    init_assgn = fill(-1, length(spikes))
    save_every = 5

    # fit regular with gibbs
    (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) = gibbs_sample!(
        regular_model, spikes, init_assgn, 100,
        0, 0.0,
        save_every, verbose=true
    )

    return true
end

@test test_gibbs_for_crashes()
