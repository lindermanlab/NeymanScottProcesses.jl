using NeymanScottProcesses
using Plots
using Random: seed!

seed!(12345)

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
max_temp = 6.0
temps = exp10.(vcat( range(max_temp, 0, length=10), fill(0.0, 3) ))


function cables_annealer(priors::CablesPriors, T)
    # Anneal cluster amplitude
    μ, σ2 = mean(priors.cluster_amplitude), var(priors.cluster_amplitude)
    new_mean = 1 + (1/max_temp) * (max_temp - log10(T)) * (μ - 1)
    priors.cluster_amplitude = specify_gamma(new_mean, σ2)

    # Anneal cluster rate
    # new_cluster_rate = T * priors.cluster_rate
    # priors.cluster_rate = new_cluster_rate

    # Anneal background amplitude
    # new_mean = (1/T) * mean(priors.bkgd_amplitude)
    # new_var = (1/T^2) * var(priors.bkgd_amplitude)
    # priors.bkgd_amplitude = specify_gamma(new_mean, new_var)

    return priors
end



# Create model for inference
priors = deepcopy(gen_priors)
model = CablesModel(T_max, priors; max_radius=max_cluster_radius)

# Construct base sampler
base_sampler = GibbsSampler(num_samples=20, save_interval=1, save_keys=(:log_p, :assignments))

# Construct annealer
sampler = Annealer(true, temps, cables_annealer, base_sampler)

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

p21 = plot(Δω_hist, ylabel="Δω", legend=false)
p22 = plot(results.log_p, ylabel="log like", legend=false)
p23 = plot(K_hist, ylabel="num clusters", legend=false)
p2 = plot(p21, p22, p23, layout=(3, 1))
display(p2)

p31 = heatmap(
    cooccupancy_matrix(assignments);
    title="true partition", c=:binary, colorbar=false
)
p32 = heatmap(
    cooccupancy_matrix(results.assignments[end-39:end]);
    title="last 40 samples", c=:binary, colorbar=false
)
p3 = plot(p31, p32, layout=(1, 2))
display(p3)

println("\007")
Δω_hist[1:10]



