### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 50f1ecc2-435c-11ed-216f-ffba4db8d631
begin
    using Pkg; Pkg.activate(".")
	using Random, LinearAlgebra
	using Plots, StatsPlots
	
	using Revise
	using NeymanScottProcesses
end

# ╔═╡ e984833f-6c15-497e-ae88-8ecd6829ff3a
using Profile

# ╔═╡ b5c55535-c171-4e44-ba68-e79938ef70e1
theme(
	:default, 
	label=nothing,
	tickfont=font(:Times, 8), 
	guidefont=(:Times, 8), 
	titlefont=(:Times, 8), 
	legendfont=(:Times, 8),
	lw=2, ms=4, msw=0.25, colorbar=false
)

# ╔═╡ 7f3bf053-62e6-4183-b537-4d7b24cc50bb
md"""
## Sample Data
"""

# ╔═╡ f8874a43-f006-4480-a9f1-94b5a6875090
begin	
	dim = 2  # Dimension of the data
	bounds = Tuple(1.0 for _ in 1:dim)  # Model bounds
	max_cluster_radius = 0.25
	
	η = 10.0  # Cluster rate
	Ak = specify_gamma(30.0, 10.0^2)  # Cluster amplitude
	A0 = specify_gamma(0.1, 1.0^2)  # Background amplitude
	
	Ψ = 5e-3 * I(dim)  # Covariance scale
	ν = 5.0  # Covariance degrees of freedom
end;

# ╔═╡ d8287c8d-f0a3-4d2c-83aa-cdca293985eb
priors = GaussianPriors(η, Ak, A0, Ψ, ν);

# ╔═╡ 12f14e36-da12-407a-92b8-33d8fdeb69c1
Random.seed!(1); gen_model = GaussianNeymanScottModel(bounds, priors);

# ╔═╡ 4eaf4c65-0247-47e6-be04-a24451d9ceba
begin
	Random.seed!(6)
	data, assignments, clusters = sample(gen_model; resample_latents=true)

	data = Vector{RealObservation{2}}(data)
	num_data = length(data)

	@show length(data)
	@show length(clusters)
end;

# ╔═╡ 0d3f0c40-0581-483e-9d08-9bbef9618d38
true_num_clusters = length(unique(assignments[assignments .!= -1]))

# ╔═╡ f7712587-1fa7-4a62-93ae-03efc1f488fe
md"""
## Fit Model
"""

# ╔═╡ 3b815b66-d78c-4c57-9054-d8f65ef48633
temps = exp10.(range(0, 0, length=30))

# ╔═╡ ede2163c-4464-496b-a31e-0f211c8186d0
gibbs_sampler = GibbsSampler(num_samples=100, save_interval=1, verbose=false, num_split_merge=10, split_merge_gibbs_moves=0);

# ╔═╡ 22f1aa9b-0b10-4c79-a41f-b7f2898441cd
anneal = s -> Annealer(false, temps, :cluster_amplitude_var, s)

# ╔═╡ 8181909e-a6d8-40de-982b-cdd628c4abd6
annealed_gibbs_sampler = anneal(gibbs_sampler);

# ╔═╡ c465c00a-30a2-4797-9b5d-2892f44bd1c6
num_data

# ╔═╡ 65f36acc-e0cf-41e8-a5f7-7d0d300e457c
begin
	Random.seed!(5)
	model = GaussianNeymanScottModel(bounds, priors)
	@time results = annealed_gibbs_sampler(model, data,
		initial_assignments=rand(1:num_data, num_data)
	)
end;

# ╔═╡ bf5f6554-0bd5-4d9e-a6c7-df17f6b8c425
model.bounds

# ╔═╡ 2c14cea8-3e6f-4eb4-b54c-362416302f23
# begin
# 	Profile.clear()
# 	@profile annealed_gibbs_sampler(model, data)
# end

# ╔═╡ e8b75f9a-a4d6-4fe1-b9db-618d8a112a6f
md"""
## Appendix: Plotting Utils
"""

# ╔═╡ a6d05df3-67b3-47bb-b513-85cb921f5e83
function plot_clusters!(plt, clusters)
	for C in clusters
		covellipse!(
			plt, C.sampled_position, C.sampled_covariance, 
			n_std=3, aspect_ratio=1, 
			alpha=0.3, c=1
		)
	end
	return plt
end

# ╔═╡ 2802010f-6b06-4eb0-ab36-beace589b687
function make_data_plot(data)
	data_x = [x.position[1] for x in data]
	data_y = [x.position[2] for x in data]
	
	plt = plot(xticks=nothing, yticks=nothing, xlim=(0, 1), ylim=(0, 1),
	frame=:box)
	scatter!(data_x, data_y, c=:black, ms=1.5, alpha=0.5)
	return plt
end

# ╔═╡ 65b5cea7-a338-4a5e-a8b6-e35de46b2bb1
pltA = let
	plt = make_data_plot(data)
	plot_clusters!(plt, clusters)
	plot!(title="True (NSP)", size=(300, 300))
end

# ╔═╡ 1e3a5cf5-9072-45c2-bb3b-46ffff01a926
let
	z = last(results.assignments)
	fit_clusters = [model.clusters[k] for k in unique(z[z .!= -1])]

	@show length(unique(z[z .!= -1]))
	pltB = make_data_plot(data)
	plot_clusters!(pltB, fit_clusters)
	plot!(title="Fit (NSP)", size=(300, 300))

	plot(pltA, pltB, size=(600, 300))
end

# ╔═╡ d9607159-cc78-431b-9189-59ad706f6982
get_num_clusters(r::NamedTuple) = [
	length(unique(r.assignments[k][r.assignments[k] .!= -1])) 
	for k in 1:length(r.assignments)
]

# ╔═╡ 7e091c40-301b-4dfb-bec5-773b7b5dd2ee
plot(
	plot(get_runtime(results), results.log_p), 
	hline!(
		plot(get_runtime(results), get_num_clusters(results)), 
		[true_num_clusters],
		ylim=(0, 2*true_num_clusters),
	),
	size=(600, 200),
)

# ╔═╡ Cell order:
# ╠═50f1ecc2-435c-11ed-216f-ffba4db8d631
# ╠═b5c55535-c171-4e44-ba68-e79938ef70e1
# ╟─7f3bf053-62e6-4183-b537-4d7b24cc50bb
# ╟─65b5cea7-a338-4a5e-a8b6-e35de46b2bb1
# ╠═f8874a43-f006-4480-a9f1-94b5a6875090
# ╠═d8287c8d-f0a3-4d2c-83aa-cdca293985eb
# ╠═12f14e36-da12-407a-92b8-33d8fdeb69c1
# ╠═4eaf4c65-0247-47e6-be04-a24451d9ceba
# ╟─0d3f0c40-0581-483e-9d08-9bbef9618d38
# ╟─f7712587-1fa7-4a62-93ae-03efc1f488fe
# ╠═bf5f6554-0bd5-4d9e-a6c7-df17f6b8c425
# ╠═3b815b66-d78c-4c57-9054-d8f65ef48633
# ╠═ede2163c-4464-496b-a31e-0f211c8186d0
# ╠═22f1aa9b-0b10-4c79-a41f-b7f2898441cd
# ╠═8181909e-a6d8-40de-982b-cdd628c4abd6
# ╠═c465c00a-30a2-4797-9b5d-2892f44bd1c6
# ╠═65f36acc-e0cf-41e8-a5f7-7d0d300e457c
# ╠═7e091c40-301b-4dfb-bec5-773b7b5dd2ee
# ╟─1e3a5cf5-9072-45c2-bb3b-46ffff01a926
# ╠═e984833f-6c15-497e-ae88-8ecd6829ff3a
# ╠═2c14cea8-3e6f-4eb4-b54c-362416302f23
# ╟─e8b75f9a-a4d6-4fe1-b9db-618d8a112a6f
# ╟─a6d05df3-67b3-47bb-b513-85cb921f5e83
# ╟─2802010f-6b06-4eb0-ab36-beace589b687
# ╟─d9607159-cc78-431b-9189-59ad706f6982
