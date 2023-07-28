using LinearAlgebra
using Plots
using PointProcessSequences
using Random
using JLD: @save

include("config.jl")  # datadir, config

Random.seed!(config[:dataseed])

bounds = config[:bounds]
R̃ = config[:max_radius]
priors = GaussianPriors(
    config[:event_rate],
    config[:event_amplitude],
    config[:bkgd_amplitude],
    config[:covariance_scale],
    config[:covariance_df],
)

model = GaussianNeymanScottModel(bounds, R̃, priors)
data, assignments, events = sample(model, resample_globals=false, resample_latents=true)
dataset = Dict(:data => data, :assignments => assignments, :events => events)

# Save
datafile = datadir * config[:datafile]
@save datafile dataset

# Plot
colormap(i) = (i == -1) ? "black" : i
scatter(
    [p.position[1] for p in data], 
    [p.position[2] for p in data],
    c=colormap.(assignments),
    legend=false,
)

