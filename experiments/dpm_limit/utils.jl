using NeymanScottProcesses
using Plots
using LinearAlgebra: I
using Random: seed!
using JLD
using Plots

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


function create_data(model)
end

function fit_data(model)
end