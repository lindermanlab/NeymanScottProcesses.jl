### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ acc93e48-e012-11eb-048e-35c010d6acee
begin
	using Pkg; Pkg.activate()
	using Plots
	using Random, StatsPlots
	
	using NeymanScottProcesses
	
	using LinearAlgebra: I
	using Random: seed!
end

# ╔═╡ 5de272b0-931a-4851-86ac-4249860a9922
theme(
	:default, label=nothing, 
	tickfont=font(:Times, 8), guidefont=(:Times, 8), titlefont=(:Times, 8), 
	legendfont=(:Times, 8),
	ms=4, msw=0.25, colorbar=false
)

# ╔═╡ be352c60-88b9-421a-8fd7-7d34e19665e6
function save_and_show(plt, label)
	savefig(plt, joinpath("../../figures/", "nsp_comparison_"*label*".png"))
	savefig(plt, joinpath("../../figures/", "nsp_comparison_"*label*".pdf"))
	return plt
end

# ╔═╡ d1d53b74-3e7b-44eb-b0e7-1e4b612edeb2
md"""
## Set parameters
"""

# ╔═╡ b0114a0b-58e4-44d2-82db-3a6131435b32
begin
	num_chains = 3
	
	dim = 2  # Dimension of the data
	bounds = Tuple(1.0 for _ in 1:dim)  # Model bounds
	max_cluster_radius = 0.25
	
	η = 10.0  # Cluster rate
	Ak = specify_gamma(30.0, 10.0^2)  # Cluster amplitude
	A0 = specify_gamma(0.1, 1.0^2)  # Background amplitude
	
	Ψ = 1e-3 * I(dim)  # Covariance scale
	ν = 5.0  # Covariance degrees of freedom
end;

# ╔═╡ b9ee61fa-c387-404b-b273-11dcfa8b63a0
md"""
## Generate data
"""

# ╔═╡ aa7eb7d8-2201-415f-b1a1-e475d5f73844
priors = GaussianPriors(η, Ak, A0, Ψ, ν);

# ╔═╡ b2b36cde-f6ed-467c-90fe-310dd1105dd1
begin
	Random.seed!(1)
	gen_model = GaussianNeymanScottModel(bounds, priors)
end;

# ╔═╡ 24098a10-f1cd-49a2-9e6e-a9502b86a731
begin
	Random.seed!(2)  # Cool seeds: 2, 24
	data, assignments, clusters = sample(gen_model; resample_latents=true)
end;

# ╔═╡ 10924047-b95e-41db-86ef-6f5c2cbea1a5
length(data)

# ╔═╡ d98caa8b-0c20-4b41-b3e2-404061a6f575
md"""
## Fit data with NSP and DPM
"""

# ╔═╡ 60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# Construct samplers
begin
	base_sampler = GibbsSampler(num_samples=100, save_interval=1, verbose=false)
    sampler = Annealer(base_sampler, 1e4, :cluster_amplitude_var; 
		num_samples=10, verbose=false)
end;

# ╔═╡ 5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
md"""
#### NSP
"""

# ╔═╡ 0a91fe9a-1f4e-4b10-beef-896d41fcadd3
begin
	nsp_model = []
	r_nsp = []
	
	t_nsp = @elapsed for chain in 1:num_chains
		Random.seed!(2 + chain)
		model = GaussianNeymanScottModel(bounds, priors)
		@show NeymanScottProcesses.log_like(model, data)
		
		r = sampler(model, data)
		
		push!(nsp_model, model)
		push!(r_nsp, r)
	end
	
	"Fit $num_chains models in $t_nsp seconds"
end

# ╔═╡ 1401a242-3e07-4d2d-be8b-ec5599868457
md"""
#### NSP with RJMCMC
"""

# ╔═╡ 7560d033-9b51-4812-a857-19862b1767ec
# Construct samplers
begin
	rj_base_sampler = ReversibleJumpSampler(num_samples=100, birth_prob=0.5)
    rj_sampler = Annealer(rj_base_sampler, 1e4, :cluster_amplitude_var; 
		num_samples=50, verbose=false)
end;

# ╔═╡ a274e4f7-b8d7-4c8a-a730-70889b0126ba
begin
	rj_nsp_model = []
	rj_r_nsp = []
	
	t_rj = @elapsed for chain in 1:num_chains
		Random.seed!(2 + chain)
		model = GaussianNeymanScottModel(bounds, priors)
		@show NeymanScottProcesses.log_like(model, data)
		
		r = rj_sampler(model, data)
		
		push!(rj_nsp_model, model)
		push!(rj_r_nsp, r)
	end
	
	"Fit $num_chains models in $t_rj seconds"
end

# ╔═╡ 8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
let
	plot(size=(250, 200), dpi=300)
	plot!([append!([0.0], r.log_p) for r in r_nsp], 
		lw=2, c=1, label=["collapsed gibbs" nothing nothing], alpha=0.7)
	plot!([append!([0.0], r.log_p) for r in rj_r_nsp], 
		lw=2, c=2, label=["reversible jump" nothing nothing], alpha=0.7)
	
	
	plot!(ylim=(1000, Inf), legend=:bottomright, ylabel="log likelihood", xlabel="number of samples")
end

# ╔═╡ b4180fdc-b209-414e-a028-b7890e69c302
plot([r.log_p for r in r_nsp], lw=3, size=(250, 200))

# ╔═╡ b301bb90-2178-4d49-bca2-e1f7ce59975f
function make_consistent(ω, data)
	# Sort labels by average position
	labels = unique(ω)
	sort!(labels, by=k -> mean(data[findall(==(k), ω)]))
	
	new_ω = zeros(Int, length(ω))
	for new_k in 1:length(labels)
		new_ω[findall(==(labels[new_k]), ω)] .= new_k
	end
	
	return new_ω
end

# ╔═╡ 31444176-7908-4eef-865d-4096aed328cd
begin
	# Make points easy to plot
	data_x = [x.position[1] for x in data]
	data_y = [x.position[2] for x in data]
	
	# Format plot for data
	make_data_plot() = plot(xticks=nothing, yticks=nothing, xlim=(0, 1), ylim=(0, 1),
		frame=:box)
	
	plt_true_data = make_data_plot()
	true_ω = make_consistent(assignments, data_x)
	scatter!(data_x, data_y, c=true_ω, title="true data")
end

# ╔═╡ 27a3553e-9211-45b8-b963-55c4511e6917
begin
	plt_fit_data_nsp = make_data_plot()
	nsp_ω = make_consistent(r_nsp[2].assignments[end], data_x)
	scatter!(data_x, data_y, c=nsp_ω, title="fit with nsp")
	
	plot(plt_true_data, plt_fit_data_nsp, size=(500, 200))
end

# ╔═╡ 65fbe8af-0c79-4710-a7a2-9b5b1b456a42
md"""
#### DPM
"""

# ╔═╡ 52a0caa9-ca84-401b-bddf-c3398ffa9bf4
begin
	Random.seed!(4)
	
	scaling = 1e6
	
	# Reset priors 
	dpm_Ak = RateGamma(Ak.α / scaling, Ak.β)
	dpm_η = η * scaling
	dpm_priors = GaussianPriors(dpm_η, dpm_Ak, A0, Ψ, ν)
	
	dpm_model = []
	r_dpm = []
	
	t_dpm = @elapsed for chain in 1:num_chains
		model = GaussianNeymanScottModel(bounds, dpm_priors)
		r = sampler(model, data)
		
		push!(dpm_model, model)
		push!(r_dpm, r)
	end
	
	"Fit $num_chains models in $t_dpm seconds"
end

# ╔═╡ 53229684-46ec-49ac-9c03-78bcaa636165
begin
	plt_ll = plot(size=(200, 200), legend=:bottomright, title="log probability")
	plot!(xlabel="sample")
	plot!([r.log_p for r in r_nsp], lw=0.5, label=["nsp" "" ""], color=4, alpha=0.75)
	plot!([r.log_p for r in r_dpm], lw=0.5, label=["dpm" "" ""], color=5, alpha=0.75)
	
	save_and_show(plt_ll, "log_prob")
end

# ╔═╡ 17406fcc-ff2e-4fef-a552-06063ec70872
begin
	plt_fit_data_dpm = make_data_plot()
	dpm_ω = make_consistent(r_dpm[1].assignments[end], data_x)
	scatter!(data_x, data_y, c=dpm_ω, title="fit with dpm")
	
	# Plot data and fits together
	plt_data = plot(plt_true_data, plt_fit_data_nsp, plt_fit_data_dpm, layout=(1, 3))
	plot!(size=(600, 200))
	
	save_and_show(plt_data, "data")
end

# ╔═╡ 77f2fb41-c0bc-4aac-a98d-0048c7b2a8b0
md"""
## Compare summary statistics

Plot the accuracy and the number of clusters.
"""

# ╔═╡ e6901d7d-6f3f-493e-9d00-7524475c5ccb
begin
	num_samples = 50
end;

# ╔═╡ e571a594-4a6e-4dd5-8762-f7330b3707ce
begin
	Random.seed!(11)
	
	num_cluster_true = length(clusters)
	
	# Set up plot
	plt_num_clusters = plot(title="number of clusters")
	plot!(xticks=(1:2, ["nsp", "dpm"]))
	plot!(size=(200, 200))
	plot!(ylim=(0, Inf))
	plot!(legend=:topleft)

	
	# Plot true number of clusters
	hline!(0.5:0.01:2.5, [num_cluster_true], lw=3, color=3, label="true")
	
		
	# Plot violin plots for NSP and DPM
	for (ind, r) in enumerate([r_nsp, r_dpm])
		num_cluster = append!([[
			length(unique(ω)) - 1 for ω in r[chain].assignments[end-num_samples+1:end]
		] for chain in 1:num_chains]...)

		num_clus_x = fill(ind, num_chains*num_samples)
		
		if ind == 2
			violin!(num_clus_x, num_cluster, color=1)
			boxplot!(num_clus_x, num_cluster, fillalpha=0.5, color=2)
		end
		
		dotplot!(num_clus_x, num_cluster, marker=(:Black, 2))
	end	
	
	# Save
	save_and_show(plt_num_clusters, "num_clusters")
end

# ╔═╡ f3ae22fc-0bb6-4469-ac9d-2bc32252c1a3
begin
	cm_true = cooccupancy_matrix(assignments)
	cm_nsp = cooccupancy_matrix(r_nsp[1].assignments[end-num_samples+1:end])
	cm_dpm = cooccupancy_matrix(r_dpm[1].assignments[end-num_samples+1:end])

	plot(
		heatmap(cm_true, title="true", ticks=nothing), 
		heatmap(cm_nsp, title="nsp", ticks=nothing),
		heatmap(cm_dpm, title="dpm", ticks=nothing),
		layout=(1, 3),
		size=(650, 200)
	)
end

# ╔═╡ cd0c0109-0dee-4da8-b9c0-282ec378bf63
begin
	Random.seed!(11)

	acc_nsp = append!([[
		1 - sum(abs, cm_true - cooccupancy_matrix(ω)) / length(cm_true)
		for ω in r_nsp[chain].assignments[end-num_samples+1:end]
	] for chain in 1:num_chains]...)
	acc_dpm = append!([[
		1 - sum(abs, cm_true - cooccupancy_matrix(ω)) / length(cm_true)
		for ω in r_dpm[chain].assignments[end-num_samples+1:end]
	] for chain in 1:num_chains]...)
	
	acc_x = repeat([1,2], inner=num_chains*50)
	acc_y = [acc_nsp; acc_dpm]
	
	# Set up plot
	plt_acc = plot(title="co-occupancy accuracy", xticks=(1:2, ["nsp", "dpm"]))
	plot!(size=(200, 200))
	
	# Plot data
	violin!(acc_x, acc_y)
	boxplot!(acc_x, acc_y, fillalpha=0.5)
	dotplot!(acc_x, acc_y, marker=(:Black, 2))
	
	save_and_show(plt_acc, "accuracy")
end

# ╔═╡ efc4144b-3578-4c91-9575-04a8f98b6816
l = @layout [
    a{0.5h} 
	[b{0.3w} c{0.3w} d{0.3w}]
]

# ╔═╡ 4fb64ab7-b329-451c-b4ff-9ae80ff6ae59
begin
	plt_everything = plot(
		plt_true_data, plt_fit_data_nsp, plt_fit_data_dpm, 
		plt_ll, plt_acc, plt_num_clusters, 
		layout=(2, 3), size=(600, 400), dpi=100
	)
	
	save_and_show(plt_everything, "full")
end

# ╔═╡ Cell order:
# ╠═acc93e48-e012-11eb-048e-35c010d6acee
# ╠═5de272b0-931a-4851-86ac-4249860a9922
# ╠═be352c60-88b9-421a-8fd7-7d34e19665e6
# ╟─d1d53b74-3e7b-44eb-b0e7-1e4b612edeb2
# ╠═b0114a0b-58e4-44d2-82db-3a6131435b32
# ╟─b9ee61fa-c387-404b-b273-11dcfa8b63a0
# ╠═aa7eb7d8-2201-415f-b1a1-e475d5f73844
# ╠═b2b36cde-f6ed-467c-90fe-310dd1105dd1
# ╠═24098a10-f1cd-49a2-9e6e-a9502b86a731
# ╠═10924047-b95e-41db-86ef-6f5c2cbea1a5
# ╠═31444176-7908-4eef-865d-4096aed328cd
# ╟─d98caa8b-0c20-4b41-b3e2-404061a6f575
# ╠═60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# ╟─5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
# ╠═0a91fe9a-1f4e-4b10-beef-896d41fcadd3
# ╟─1401a242-3e07-4d2d-be8b-ec5599868457
# ╠═7560d033-9b51-4812-a857-19862b1767ec
# ╠═a274e4f7-b8d7-4c8a-a730-70889b0126ba
# ╠═8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
# ╠═b4180fdc-b209-414e-a028-b7890e69c302
# ╠═b301bb90-2178-4d49-bca2-e1f7ce59975f
# ╠═27a3553e-9211-45b8-b963-55c4511e6917
# ╟─65fbe8af-0c79-4710-a7a2-9b5b1b456a42
# ╠═52a0caa9-ca84-401b-bddf-c3398ffa9bf4
# ╠═53229684-46ec-49ac-9c03-78bcaa636165
# ╠═17406fcc-ff2e-4fef-a552-06063ec70872
# ╟─77f2fb41-c0bc-4aac-a98d-0048c7b2a8b0
# ╠═e6901d7d-6f3f-493e-9d00-7524475c5ccb
# ╠═e571a594-4a6e-4dd5-8762-f7330b3707ce
# ╠═f3ae22fc-0bb6-4469-ac9d-2bc32252c1a3
# ╠═cd0c0109-0dee-4da8-b9c0-282ec378bf63
# ╠═efc4144b-3578-4c91-9575-04a8f98b6816
# ╠═4fb64ab7-b329-451c-b4ff-9ae80ff6ae59
