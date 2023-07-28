import Random
import PyPlot: plt

using LinearAlgebra
using PointProcessSequences

Random.seed!(1)

# Specify parameters
bounds = (1.0, 1.0)
area = bounds[1] * bounds[2]
max_radius = Inf

event_rate = 5.0 # events ≈ 5
event_amplitude_mean = 30.0
event_amplitude_var = 1e-1
bkgd_amplitude_mean = 10.0
bkgd_amplitude_var = 1e-10
covariance_scale = [1.0 0.0; 0.0 1.0] * 1e-3
covariance_df = 10.0

event_amplitude = specify_gamma(event_amplitude_mean, event_amplitude_var)
bkgd_amplitude = specify_gamma(bkgd_amplitude_mean, bkgd_amplitude_var)  

# Generate empty model
model = GaussianNeymanScottModel(
    bounds,
    max_radius,

    event_rate,
    event_amplitude,
    bkgd_amplitude,
    covariance_scale,
    covariance_df
)

# Sample data
data, assignments, events = 
    sample(model, resample_globals=true, resample_latents=true)

# Plot the data
colors = ["blue", "red", "green", "orange"]
colormap(i) = i == -1 ? "k" : colors[((i-1)%length(colors)) + 1]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("truth")
plt.scatter(
    [p.position[1] for p in data], 
    [p.position[2] for p in data],
    c=colormap.(assignments),
    s=20,
    lw=0
)

# Create a copy of the model (with different globals)
model = deepcopy(model)

# Set the model to have a much larger event amplitude variance
# In other words, make the prior on event amplitude notably weaker
model.priors.event_amplitude = specify_gamma(
    event_amplitude_mean, event_amplitude_mean^2
)
PointProcessSequences._gibbs_reset_model_probs(model)

# Display relative probabilities for debugging
@show model.bkgd_log_prob, model.new_cluster_log_prob

# Fit the model!!
(
    assignments,
    assignment_hist,
    log_p_hist,
    latent_event_hist,
    globals_hist
) = annealed_gibbs!(
    model,
    data,
    fill(-1, length(data)),
    5,  # anneal rounds
    500,  # samples per anneal
    1000.0,  # max temp
    0,
    5.0,
    50;
    verbose=true,
    anneal_background=true
)

# Plot resulting assignments
plt.subplot(1, 2, 2)
plt.title("est")
plt.scatter(
    [p.position[1] for p in data], 
    [p.position[2] for p in data],
    c=colormap.(assignments),
    s=20,
    lw=0
)

# Plot cluster centroids
events = latent_event_hist[end]
plt.scatter(
    [e.position[1] for e in events],
    [e.position[2] for e in events],
    s=20,
    marker="x",
    c=colormap.([e.index for e in events])
)