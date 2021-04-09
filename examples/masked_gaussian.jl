using NeymanScottProcesses
using PyPlot: plt
using Random: seed!

seed!(1234)

# Generate a synthetic dataset.
(
    model,
    heldout_region,
    heldout_data,
    observed_data,
    heldout_assignments,
    observed_assignments,
    true_clusters
) = NeymanScottProcesses.masked_gaussian_2d_data();

all_assignments = vcat(
    observed_assignments,
    heldout_assignments
)
all_datapoints = vcat(
    observed_data,
    heldout_data
)


function cluster_colors(assgn::Int64)
    palette = ["r", "g", "b"]
    if assgn < 0
        return "k"
    else
        return palette[1 + (assgn % length(palette))]
    end
end

# Clustered observations.
scatter_kwargs = Dict(:lw => 0, :s => 10)
fig, ax = plt.subplots(1, 1)
ax.scatter(
    [d[1] for d in all_datapoints],
    [d[2] for d in all_datapoints],
    c=[cluster_colors(i) for i in all_assignments];
    scatter_kwargs...
)

fig, ax = plt.subplots(1, 1)
plot!(ax, heldout_region; color="k", alpha=0.3, lw=0, zorder=-1)
ox = [d[1] for d in observed_data]
oy = [d[2] for d in observed_data]
hx = [d[1] for d in heldout_data]
hy = [d[2] for d in heldout_data]
ax.scatter(ox, oy, color="k"; scatter_kwargs...)
ax.scatter(hx, hy, color="r"; scatter_kwargs...)


# ===
# INFERENCE
# ===

# Create model for inference
est_model = NeymanScottModel(
    deepcopy(model.domain),
    deepcopy(model.priors)
)

# Construct sampler
base_sampler = GibbsSampler(
    num_samples=100,
    save_interval=10
)
masked_sampler = MaskedSampler(
    base_sampler,
    heldout_region;
    heldout_data=heldout_data,
    num_samples=10
)
sampler = AnnealedSampler(
    masked_sampler,
    10.0,
    :bkgd;
    num_samples=10
)

# Run sampler
results = base_sampler(
    model, deepcopy(all_datapoints);
    initial_assignments=deepcopy(all_assignments)
)
sampled_data, sampled_assignments = sample_in_mask(model, heldout_region)


ox = [d[1] for d in observed_data]
oy = [d[2] for d in observed_data]
sx = [d[1] for d in sampled_data]
sy = [d[2] for d in sampled_data]

fig, ax = plt.subplots(1, 1)
ax.scatter(ox, oy, color="k", lw=0, s=10)
ax.scatter(sx, sy, color="r", lw=0, s=10)
plot!(ax, heldout_region; color="k", alpha=0.3, lw=0, zorder=-1)

plt.show()
