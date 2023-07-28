import PyPlot: plt
using PointProcessSequences

include("./../tests/util.jl")


# generate data and model
model, _ = make_test_spikes_and_model(max_time=10.0)
spikes, _, _ = sample(model, resample_latents=true)
@show length(spikes)

# generate distributed model
regular_model = deepcopy(model)
distr_model = DistributedPPSeq(deepcopy(model), 4)

# set parameters
init_assgn = fill(-1, length(spikes))
num_anneals = 5
samples_per_anneal = 500
max_temp = 10.0^2
save_every = 1

# fit regular model
(
    assignments,
    assignment_hist,
    log_p_hist,
    latent_event_hist,
    globals_hist
) = annealed_gibbs!(
    regular_model, spikes, init_assgn,
    num_anneals, samples_per_anneal, max_temp,
    0, 0.0,
    save_every, verbose=true
)

# fit distributed model
(
    distr_assignments,
    distr_assignment_hist,
    distr_log_p_hist,
    distr_latent_event_hist,
    distr_globals_hist
) = annealed_gibbs!(
    distr_model, spikes, init_assgn,
    num_anneals, samples_per_anneal, max_temp,
    0, 0.0,
    save_every, verbose=true
)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(log_p_hist, label="regular")
plt.plot(distr_log_p_hist, label="distributed")

plt.subplot(2, 1, 2)
plt.plot([length(e) for e in latent_event_hist], label="regular")
plt.plot([length(e) for e in distr_latent_event_hist], label="distributed")
plt.legend()