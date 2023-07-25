using Revise

using PointProcessSequences
using Random
using JLD
using Plots

include("./config.jl")

Random.seed!(config[:mask_seed])

dataset = load(datadir * config[:datafile], "dataset")
data = dataset[:data]

# Construct model
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

# Insert masks
mask_radius = 0.05
percent_masked = 0.30
masks = create_random_mask(model, mask_radius, percent_masked)

unmasked_data = filter(x -> !(x in masks), data)
masked_data = filter(x -> x in masks, data)

# Plot the masks
function circle_shape(x, r)
    θ = LinRange(0, 2π, 100)
    return x[1] .+ r*cos.(θ), x[2] .+ r*sin.(θ)
end
c1(x) = x.position[1]
c2(x) = x.position[2]
color(k) = (k == -1) ? "black" : k

p1 = plot(xlim=(0, 1), ylim=(0, 1), size=(500, 500), title="true", legend=:bottomright)
for m in masks
    plot!(
        p1, circle_shape(m.center, m.radius), 
        seriestype=:shape, color="yellow", fillalpha=0.2,
        label=nothing,
    )
end
scatter!(p1, c1.(unmasked_data), c2.(unmasked_data), label="observed")
scatter!(p1, c1.(masked_data), c2.(masked_data), color="yellow", label="masked")
display(p1)


ω, ω_hist, logp_hist, test_logp_hist, E_hist, G_hist = masked_gibbs!(
    model,
    masked_data,
    unmasked_data,
    masks,
    fill(-1, length(unmasked_data)),
    10,
    500,
    0,
    1.0,
    1;
    verbose=true
)

# Plot results
sampled_data, sampled_ω = sample_masked_data(model, masks)

p2 = scatter(
    c1.(sampled_data), c2.(sampled_data), 
    color="yellow", 
    label="imputed", 
    title="predicted",
    legend=:bottomright,
)
scatter!(p2, c1.(unmasked_data), c2.(unmasked_data), color="black", label="observed")

p = plot(p1, p2, layout=(1, 2), size=(800, 400))
display(p)


# # Save results
# results = Dict(
#     :assignments => ω,
#     :assignment_hist => ω_hist,
#     :log_p_hist => logp_hist,
#     :event_hist => E_hist,
#     :globals_hist => G_hist,
#     :model => model,
# )
# savepath = datadir * config[:run_resultsfile]
# @save savepath results
