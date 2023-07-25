using PointProcessSequences
using Random
using JLD

include("./config.jl")

Random.seed!(config[:run_seed])

dataset = load(datadir * config[:datafile], "dataset")
data = dataset[:data]

# Fit model
bounds = config[:bounds]
R̃ = config[:max_radius]
A = config[:event_amplitude]

priors = GaussianPriors(
    config[:event_rate],
    specify_gamma(mean(A), mean(A)^2),
    config[:bkgd_amplitude],
    config[:covariance_scale],
    config[:covariance_df],
)

model = GaussianNeymanScottModel(bounds, R̃, priors)
ω, ω_hist, logp_hist, E_hist, G_hist = annealed_gibbs!(
    model, data, fill(-1, length(data));
    num_anneals=5,
    samples_per_anneal=500,
    max_temperature=1000.0,
    verbose=true,
    anneal_background=true
)

# Save results
results = Dict(
    :assignments => ω,
    :assignment_hist => ω_hist,
    :log_p_hist => logp_hist,
    :event_hist => E_hist,
    :globals_hist => G_hist,
    :model => model,
)
savepath = datadir * config[:run_resultsfile]
@save savepath results
