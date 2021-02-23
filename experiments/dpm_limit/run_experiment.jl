using NeymanScottProcesses
using Plots
using LinearAlgebra: I
using Random: seed!
using JLD

seed!(1234)


# ===
# SCRIPT PARAMETERS
# ===

datadir = "/Users/degleris/data/cables/"
results_file = "results/dpm_limit.jld"

num_trials = 2
num_parameters = 5

θ_arr = exp10.(range(0, 2, length=num_parameters))


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
    model = GaussianNeymanScottModel(bounds, priors)

    data, assignments, clusters = sample(model; resample_latents=true)
    @show length(data)

    # Generate masks
    masks = create_random_mask(model, mask_radius, percent_masked)
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

        @show (trial_ind, param_ind)

        gen_priors, masks, masked_data, unmasked_data = 
            dataset.priors, dataset.masks, dataset.masked_data, dataset.unmasked_data
        

        # Make priors like DPM
        θ = θ_arr[param_ind]
        K_θ = θ * K
        Ak_θ = RateGamma(Ak.α / θ, Ak.β)

        @show (θ, K_θ, Ak_θ)

        # Create model for inference
        priors =  GaussianPriors(K_θ, Ak_θ, A0, Ψ, ν)
        model = GaussianNeymanScottModel(bounds, priors; max_radius=max_cluster_radius)

        # Construct sampler
        base_sampler = GibbsSampler(num_samples=50, save_interval=10, verbose=false)
        masked_sampler = MaskedSampler(base_sampler, masks; masked_data=masked_data, 
            num_samples=3, verbose=false)
        sampler = Annealer(masked_sampler, 200.0, :cluster_amplitude_var; 
            num_samples=3, verbose=false)

        # Run sampler
        @time r = sampler(model, unmasked_data)

        results = (
            priors=priors, model=model, results=r
        )

        push!(trial_results, results)

        println()
    end


    push!(results_arr, trial_results)
end

f = joinpath(datadir, results_file)
JLD.@save f dataset_arr results_arr


scores = [
    mean([results_arr[1][j].results.test_log_p[end] for i in 1:num_trials])
    for j = 1:num_parameters
]
# plot(θ_arr, scores, xlabel="K / K0", ylabel="test log like", legend=false, lw=6)