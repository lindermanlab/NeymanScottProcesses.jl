import Random

using Revise
using LinearAlgebra
using PointProcessSequences

include("utils.jl")

#function run()
Random.seed!(123)

# Specify parameters
max_time = 200.0
max_radius = Inf
num_nodes = 10
num_marks = 100

event_rate = 0.1 # events ≈ 10, cables ≈ 300
event_amplitude_mean = 30.0
event_amplitude_var = 1e-1
bkgd_amplitude_mean = 1.5
bkgd_amplitude_var = 1e-15

node_concentration = 5e-2 * ones(num_nodes)
mark_concentration = 1e-2

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
@assert (length(unique(assignments)) == length(events) + 1)

# Weaken priors and generate a simpler model
# priors.event_amplitude = specify_gamma(30.0, 5.0^2)
# priors.bkgd_mark_concentration = 100.0 * ones(num_marks, num_nodes)

model = CablesModel(max_time, max_radius, priors)
#model.globals = deepcopy(gen_model.globals)

@show model.bkgd_log_prob, model.new_cluster_log_prob

# Fit the model!!
(
    est_assignments,
    assignment_hist,
    log_p_hist,
    latent_event_hist,
    globals_hist
) = gibbs_sample!(
    model,
    data,
    fill(-1, length(data)),  #deepcopy(assignments);
    verbose=true,
    num_samples=10
)

est_events = latent_event_hist[end]

# ) = annealed_gibbs!(
#     model,
#     data,
#     fill(-1, length(data));
#     verbose=true,
#     anneal_background=true,
#     num_anneals=2,
#     samples_per_anneal=1
# )

@show sum(est_assignments .== -1)

plot_loglike(log_p_hist, latent_event_hist)
plot_cables(data, assignments, est_assgn=est_assignments)
#end
