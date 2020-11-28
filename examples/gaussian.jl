using LinearAlgebra: I
using NeymanScottProcesses
using Plots
using Random: seed!

seed!(1234)




# ===
# PARAMETERS
# ===

dim = 2  # Dimension of the data
bounds = Tuple(4.0 for _ in 1:dim)  # Model bounds

K = 4.0  # Event rate
Ak = specify_gamma(20.0, 3.0)  # Event amplitude
A0 = specify_gamma(20.0, 3.0)  # Background amplitude

Ψ = 1e-3 * I(dim)  # Covariance scale
ν = 5.0  # Covariance degrees of freedom




# ===
# GENERATIVE MODEL
# ===

gen_priors = GaussianPriors(K, Ak, A0, Ψ, ν)
gen_model = GaussianNeymanScottModel(bounds, gen_priors)

data, assignments, events = sample(gen_model; resample_latents=true)

# Visualize results
p1 = plot(data, assignments, xlim=(0, 2), ylim=(0, 2), title="")
@show length(data)
display(p1)




# ===
# INFERENCE
# ===

# Create model for inference
priors = deepcopy(gen_priors)
model = GaussianNeymanScottModel(bounds, priors)

# Construct sampler
subsampler = GibbsSampler(num_samples=100, save_interval=1)
sampler = Annealer(subsampler, 200.0, :event_amplitude_var)

# Run sampler
results = sampler(model, data)

# Visualize results
p2 = plot(data, last(results.assignments), xlim=(0, 2), ylim=(0, 2), title="estimate")
plot(p1, p2, layout=(1, 2), size=(800, 400))
