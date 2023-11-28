# Activate local environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using NeymanScottProcesses
using Plots
using LinearAlgebra: I
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

mask_radius = 0.05
percent_masked = 0.20



# ===
# GENERATIVE MODEL
# ===

gen_priors = GaussianPriors(K, Ak, A0, Ψ, ν)
gen_model = GaussianNeymanScottModel(bounds, gen_priors)

data, assignments, clusters = sample(gen_model; resample_latents=true)

# Generate masks
masks = create_random_mask(gen_model, mask_radius, percent_masked)
masked_data, unmasked_data = split_data_by_mask(data, masks)

# Visualize results
p1 = plot(data, assignments, xlim=(0, 2), ylim=(0, 2), size=(500, 500), title="")
plot!(p1, masks)

display(p1)
@show length(data)




# ===
# INFERENCE
# ===

# Create model for inference
priors = deepcopy(gen_priors)
model = GaussianNeymanScottModel(bounds, priors; max_radius=max_cluster_radius)

# Construct sampler
base_sampler = GibbsSampler(num_samples=50, save_interval=10)
masked_sampler = MaskedSampler(base_sampler, masks; masked_data=masked_data, num_samples=3)
sampler = Annealer(masked_sampler, 200.0, :cluster_amplitude_var; num_samples=3)

# Run sampler
results = sampler(model, unmasked_data)
sampled_data, sampled_assignments = sample_masked_data(model, masks)

# Visualize results
p2 = plot(
    unmasked_data, last(results.assignments);
    size=(400, 400), xlim=(0, 2), ylim=(0, 2), title="estimate"
)
plot!(p2, masks)
plot!(p2, sampled_data, color="red")

plot(p1, p2, layout=(1, 2), size=(800, 400))
