using NeymanScottProcesses
using Plots
using Random: seed!

seed!(1234)


# TODO
# - [X] Plot co-occupancy matrix (true_assign, assign)
# - [X] Plot co-occupancy correlations across samples
# - [X] Fit synthetic data, 100 vocab
# - [X] Specify inverse gamma via mean and variance
# - [X] Set up custom annealer, anneal cluster rate
# - [X] Debug how assignments are changing over time
# - [X] Debug number of clusters over time
# - [X] Explore modifying the cluster width parameters
#           Didn't do much :/
# - [X] Analyze prior / data word distributions
# - [X] Draw some samples from a Dirichlet, maybe mess with the word concentration?
# - [X] **Fit synthetic data, 1000 vocab**
#           Changing the word & embassy concentrations to 1.0 made the fit work

# - [ ] Push to github
# - [ ] Debug 10k words
# - [ ] Debug 1k words
# - [ ] Go through Alex's PR
    # - [ ] Figure out how to merge cables model with Alex...
# - [ ] **Fit synthetic data, 10k vocab**


# - [ ] Debug background log like / new cluster log like / current log like
# - [ ] Speed up code (ideally 10x or so)
# - [ ] Debug log likelihood
# - [ ] Add analysis functions to diagnostic.jl

# - [ ] Parse real data
# - [ ] Obtain a reasonable fit on the real data


# - ??? Why are the annealing parameters so insane?
#           A: It's overkill at this point
#           A: Somewhere in the CRP probabilities we compute something of the order
#               log(T * params), which means we have to scale temperature exponentially
#               to scale the CRP probabilities linearly, causing numerical instability

# - ??? Why do we get clusters with insane (empirical) widths?
#           A: Because 1000 words is so much more important than 1 timestamp

# - ??? Why does the word concentration matter so much?


# ===
# PARAMETERS
# ===

T_max = 100.0  # Model bounds
max_cluster_radius = 30.0

word_dim = 10_000
embassy_dim = 100

K = 1 / 15.0  # Cluster rate
Ak = specify_gamma(100.0, 1.0^2)  # Cluster amplitude
σ0 = specify_inverse_gamma(2.0, 0.01^2)  # Cluster width
ϵ_conc = 1e-0 * ones(embassy_dim)  # Embassy concentration
ν_conc = 1e-0  # Word concentration

A0 = specify_gamma(10.0, 1e-1)  # Background amplitude
ϵ0_conc = ones(embassy_dim)  # Background embassy concentration
ν0_conc = ones(word_dim, embassy_dim)  # Background word concentration
δ_conc = ones(7)




# ===
# GENERATIVE MODEL
# ===

gen_priors = CablesPriors(K, Ak, σ0, ϵ_conc, ν_conc, A0, ϵ0_conc, ν0_conc, δ_conc)
gen_model = CablesModel(T_max, gen_priors)

data, assignments, clusters = sample(gen_model; resample_latents=true)


# Visualize results
p11 = plot(data, assignments, size=(250, 150))

display(p11)
println("number of cables: $(length(data))")
println("vocabulary size: $word_dim")
println()




# ===
# INFERENCE
# ===

# Custom annealing
function cables_annealer(priors::CablesPriors, T)
    # Anneal background amplitude
    new_mean = (1/T) * mean(priors.bkgd_amplitude)
    new_var = (1/T^2) * var(priors.bkgd_amplitude)
    priors.bkgd_amplitude = specify_gamma(new_mean, new_var)

    # Anneal cluster amplitude
    new_mean = 1 + (1/sqrt(T)) * (mean(priors.cluster_amplitude) - 1)
    new_var = var(priors.cluster_amplitude)
    priors.cluster_amplitude = specify_gamma(new_mean, new_var)

    # Anneal cluster rate
    # new_cluster_rate = T * priors.cluster_rate
    # priors.cluster_rate = new_cluster_rate

    return priors
end

# Create model for inference
priors = deepcopy(gen_priors)

model = CablesModel(T_max, priors; max_radius=max_cluster_radius)

# Construct sampler
base_sampler = GibbsSampler(num_samples=20, save_interval=1, save_keys=(:log_p, :assignments))
sampler = Annealer(base_sampler, 1e6, cables_annealer; num_samples=10)

# Run sampler
@time results = sampler(model, data)

# Visualize results
p12 = plot(data, last(results.assignments); title="estimate", legend=false, size=(250, 150))

p1 = plot(p11, p12, layout=(1, 2), size=(600, 150))

display(p1)


# ===
# ANALYSIS
# ===

get_delta(a1, a2) = count(x->(x != 0), a1 - a2)
get_deltas(a_hist) = [get_delta(a_hist[i], a_hist[i-1]) for i in 2:length(a_hist)]

num_clusters(ω) = length(unique(ω)) - 1

Δω_hist = get_deltas(results.assignments)
K_hist = num_clusters.(results.assignments)

println("\007")
Δω_hist[1:10]



