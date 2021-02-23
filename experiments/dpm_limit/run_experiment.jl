using NeymanScottProcesses
using Plots
using LinearAlgebra: I
using Random: seed!

seed!(1234)


# ===
# SCRIPT PARAMETERS
# ===

num_trials = 1
num_parameters = 3


# ===
# BASE MODEL PARAMETERS
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

dataset_arr = []


for trial_ind in 1:num_trials
    priors = GaussianPriors(K, Ak, A0, Ψ, ν)
    model = GaussianNeymanScottModel(bounds, gen_priors)

    data, assignments, clusters = sample(gen_model; resample_latents=true)

    # Generate masks
    masks = create_random_mask(gen_model, mask_radius, percent_masked)
    masked_data, unmasked_data = split_data_by_mask(data, masks)

    dataset = (
        priors=priors, model=model, data=data, assignments=assignments, 
        clusters=clusters, masks=masks, masked_data=masked_data, unmasked_data=unmasked_data
    )

    push!(dataset_arr, dataset)
end




# ===
# INFERENCE
# ===

results_arr = []

for trial_ind in 1:num_trials

    dataset = dataset_arr[trial_ind]

    trial_results = []

    for param_ind in 1:num_parameters
        
        # Create model for inference
        priors = nothing # TODO
        model = GaussianNeymanScottModel(bounds, priors; max_radius=max_cluster_radius)

        # Construct sampler
        base_sampler = GibbsSampler(num_samples=50, save_interval=10)
        masked_sampler = MaskedSampler(base_sampler, masks; masked_data=masked_data, num_samples=3)
        sampler = Annealer(masked_sampler, 200.0, :cluster_amplitude_var; num_samples=3)

        # Run sampler
        results = sampler(model, unmasked_data)

        inference_results = (
            priors=priors, model=model, results=results
        )

        push!(trial_results, inference_results)
    end


    push!(results_arr, trial_results)
end
