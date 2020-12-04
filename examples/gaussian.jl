using LinearAlgebra: I
using NeymanScottProcesses
using Plots
using Random: seed!

seed!(1234)


using Profile

# ===
# PARAMETERS
# ===

dim = 2  # Dimension of the data
bounds = Tuple(4.0 for _ in 1:dim)  # Model bounds
max_cluster_radius = 0.5

K = 4.0  # Cluster rate
Ak = specify_gamma(20.0, 3.0)  # Cluster amplitude
A0 = specify_gamma(20.0, 3.0)  # Background amplitude

Ψ = 1e-3 * I(dim)  # Covariance scale
ν = 5.0  # Covariance degrees of freedom

mask_radius = 0.1
percent_masked = 0.30



# ===
# GENERATIVE MODEL
# ===

gen_priors = GaussianPriors(K, Ak, A0, Ψ, ν)
gen_model = GaussianNeymanScottModel(bounds, gen_priors)

data, assignments, clusters = sample(gen_model; resample_latents=true)

# Generate mask
mask = create_random_mask(gen_model, mask_radius, percent_masked)
heldout_data, observed_data = split_data_by_mask(data, mask)

# Visualize results
p1 = plot(data, assignments, xlim=(0, 2), ylim=(0, 2), size=(500, 500), title="")
plot!(p1, mask)

display(p1)
@show length(data)




# ===
# INFERENCE
# ===

# Create model for inference
priors = deepcopy(gen_priors)
model = GaussianNeymanScottModel(bounds, priors; max_radius=max_cluster_radius)

# Construct sampler
base_sampler = GibbsSampler(
    num_samples=50,
    save_interval=10
)
masked_sampler = MaskedSampler(
    base_sampler,
    mask;
    heldout_data=heldout_data,
    num_samples=3
)
sampler = AnnealedSampler(
    masked_sampler,
    200.0,
    :cluster_amplitude_var;
    num_samples=3
)

# Run sampler
results = sampler(model, observed_data)
sampled_data, sampled_assignments = sample_data_in_mask(model, mask)

# Visualize results
p2 = plot(
    observed_data, last(results.assignments);
    size=(400, 400), xlim=(0, 2), ylim=(0, 2), title="estimate"
)
plot!(p2, mask)
plot!(p2, sampled_data, color="red")

plot(p1, p2, layout=(1, 2), size=(800, 400))
