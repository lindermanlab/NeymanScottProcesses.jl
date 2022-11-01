### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ acc93e48-e012-11eb-048e-35c010d6acee
begin
    using Pkg; Pkg.activate("."); Pkg.instantiate();
	using Plots
	using Random, StatsPlots
	
	using NeymanScottProcesses
	
	using LinearAlgebra: I
	using Random: seed!
end

# ╔═╡ 9a9d1185-0e58-40e5-a4b0-e5efb1f9d448
using MCMCDiagnosticTools

# ╔═╡ d35f633b-5df3-4386-84fd-8b609462ac72
using StatsBase: autocor

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

	data = Vector{RealObservation{2}}(data)
end;

# ╔═╡ 10924047-b95e-41db-86ef-6f5c2cbea1a5
length(data)

# ╔═╡ 31444176-7908-4eef-865d-4096aed328cd
begin
	# Make points easy to plot
	data_x = [x.position[1] for x in data]
	data_y = [x.position[2] for x in data]
	
	# Format plot for data
	function make_data_plot()
		plt = plot(xticks=nothing, yticks=nothing, xlim=(0, 1), ylim=(0, 1),
		frame=:box)
		scatter!(data_x, data_y, c=:black, ms=1.5, alpha=0.5)
		return plt
	end
	
	plt_true_data = make_data_plot()
	
	function plot_clusters!(plt, clusters)
		for C in clusters
			covellipse!(
				plt, C.sampled_position, C.sampled_covariance, 
				n_std=3, aspect_ratio=1, 
				alpha=0.3, c=1
			)
		end
	end

	plot_clusters!(plt_true_data, clusters)
	
	plot!(title="True (NSP)", size=(200, 200))

end

# ╔═╡ d98caa8b-0c20-4b41-b3e2-404061a6f575
md"""
## Fit data with NSP and DPM
"""

# ╔═╡ 60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# Construct samplers
begin
	base_sampler = GibbsSampler(num_samples=100, save_interval=1, verbose=false)

	temps = exp10.([range(4, 0, length=50); zeros(50)])
	sampler = Annealer(false, temps, :cluster_amplitude_var, base_sampler)
	
    #sampler = Annealer(base_sampler, 1e4, :cluster_amplitude_var; 
	#	num_samples=25, verbose=false)
end;

# ╔═╡ 5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
md"""
#### NSP
"""

# ╔═╡ f1e0e42f-c321-4969-b1cd-c0385fa73ae5
foo = let
	_base_sampler = GibbsSampler(num_samples=100, save_interval=1, verbose=false)
	
	temps = exp10.([range(4, 0, length=20); zeros(20)])
	_sampler = Annealer(false, temps, :cluster_amplitude_var, _base_sampler)
	
	_model = GaussianNeymanScottModel(bounds, priors)
	
	r = _sampler(_model, data)
	println(NeymanScottProcesses.log_like(_model, data) / length(data), "\n")

	(r=r, model=_model)
end

# ╔═╡ 6234628c-0274-4b23-9255-69e5f6878549
num_clusters(r::NamedTuple) = [length(unique(r.assignments[k][r.assignments[k] .!= -1])) for k in 1:length(r.assignments)]

# ╔═╡ db0fa5a1-11f8-44fa-a17a-814c9ffcfcf8
bkgd_rate(r::NamedTuple) = [r.globals[k].bkgd_rate for k in 1:length(r.globals)]

# ╔═╡ 058b8bb9-976a-4085-844c-1fa3fb5cb3a4
true_num_clusters = length(unique(assignments[assignments .!= -1]))

# ╔═╡ 0a91fe9a-1f4e-4b10-beef-896d41fcadd3
begin
	nsp_model = []
	r_nsp = []
	
	t_nsp = @elapsed for chain in 1:num_chains
		Random.seed!(930 + chain)
		model = GaussianNeymanScottModel(bounds, priors)
		
		r = sampler(model, data)
		println(NeymanScottProcesses.log_like(model, data) / length(data))
		
		push!(nsp_model, model)
		push!(r_nsp, r)
	end
	
	"Fit $num_chains models in $t_nsp seconds"
end

# ╔═╡ 270448bf-3eea-4ce6-87f5-f986d2e05057
let
	plt = plot(size=(400, 200))
	[plot!(num_clusters(r_nsp[k]), lw=2) for k in 1:3]
	hline!([true_num_clusters], c=:Black, lw=2, label="True")
	plt
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
		num_samples=250, verbose=false)
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

# ╔═╡ f20d5e13-172c-4955-82d5-0248ab48cad4
get_num_clusters(r) = num_clusters(r)

# ╔═╡ 8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
plt_cgbd_ll = let
	plot(size=(250, 200), dpi=200)
	
	plot!(
		[append!([0.0], r.log_p) for r in r_nsp] / length(data), 
		lw=2, 
		c=1, 
		label=["CG" nothing nothing], 
		alpha=0.7
	)
	plot!(
		[append!([0.0], r.log_p) for r in rj_r_nsp] / length(data), 
		lw=2, 
		c=2, 
		label=["RJMCMC" nothing nothing], 
		alpha=0.7
	)
	
	
	plot!(
		ylim=(1000/length(data), Inf), 
		legend=:bottomright, 
		ylabel="Mean Log Likelihood", 
		xlabel="Sample",
		xticks=(0:5000:15_000, ["0", "5k", "10k", "15k"]),
		grid=false
	)
end

# ╔═╡ 6044fecb-479f-49d2-ab76-f30cdbae0691
plt_cgbd_nc = let
	plot(size=(250, 200), dpi=200)
	plot!(
		[get_num_clusters(r) for r in r_nsp], 
		lw=2, c=1, label=nothing, alpha=0.5
	)
	plot!(
		[get_num_clusters(r) for r in rj_r_nsp], 
		lw=2, c=2, label=nothing, alpha=0.5
	)
	hline!([true_num_clusters], c=:Black, lw=2, label="True")
	
	
	plot!(
		ylabel="Number of Clusters", xlabel="Sample", legend=:bottomright,
		xticks=(0:5000:15_000, ["0", "5k", "10k", "15k"]),
		ylim=(0, 16), grid=false
	)
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

# ╔═╡ 27a3553e-9211-45b8-b963-55c4511e6917
begin
	plt_fit_data_nsp = make_data_plot()
	#nsp_ω = make_consistent(r_nsp[2].assignments[end], data_x)
	#scatter!(data_x, data_y, c=nsp_ω, title="fit with nsp")
	plot_clusters!(plt_fit_data_nsp, nsp_model[1].clusters)
	plot!(plt_fit_data_nsp, title="Learned (NSP)")

	
	
	plot(plt_true_data, plt_fit_data_nsp, size=(500, 200))
	"Started NSP plot"
end

# ╔═╡ d83f7bf8-552a-4c1f-ab09-6c394d2deb4e
md"""
## MCMC Diagnostics
"""

# ╔═╡ ed090860-0352-4a63-9675-80a80e71cbeb
make_chain(rs, f) = reshape(
	hcat([f(r) for r in rs]...), 
	:, 1, num_chains
);

# ╔═╡ b48c741d-b522-4314-8a95-ca5ff2089797
chain_num_clusters_cg = make_chain(r_nsp, num_clusters);

# ╔═╡ db3947df-abe6-4349-ba56-3d89b72d5a1a
chain_num_clusters_rj = make_chain(rj_r_nsp, num_clusters);

# ╔═╡ 4929b857-51bd-4c2d-9cf8-a0d59820f502
size(chain_num_clusters_cg)

# ╔═╡ d94cd58d-38a5-4b9e-be60-d252491af4ea
ess_method(chain) = ess_rhat(chain; method=ESSMethod())

# ╔═╡ 84d56a3d-f364-4089-8808-289dd064e2f0
get_ess(chain, samples) = [ess_method(chain[1:s, :, :])[1][1] for s in samples]

# ╔═╡ 6650bf54-e43a-4916-a0c7-1c0b0f0e2f4e
get_psr(chain, samples) = [gelmandiag(chain[1:s, :, :])[2][1] for s in samples]

# ╔═╡ ff2163c6-7cc8-47de-8acf-54295282eab6


# ╔═╡ 8f65e600-93c1-4a37-8a9e-9b187379689d
begin
	autocor(chain_num_clusters_cg[:, 1, 1])
end

# ╔═╡ 1f0061bf-2699-4e6d-bcbe-2c5fb4287d7a
let
	_x = 1000:10:size(chain_num_clusters_cg, 1)
	
	plt1 = plot(title="Effective Sample Size", legend=:topleft)
	plot!(_x, get_ess(chain_num_clusters_cg, _x), lw=2, label="CG")
	plot!(_x, get_ess(chain_num_clusters_rj, _x), lw=2, label="RJ")

	plt2 = plot(title="Potential Scale Reduction")
	plot!(_x, get_psr(chain_num_clusters_cg, _x), lw=2)
	plot!(_x, get_psr(chain_num_clusters_rj, _x), lw=2)

	plot(plt1, plt2, layout=(2, 1))
	
end

# ╔═╡ 4a008c9b-fbc4-407c-91cf-d5ca9e0a4662
minimum(get_psr(chain_num_clusters_cg, 1000:10:size(chain_num_clusters_cg, 1)))

# ╔═╡ 69e40e4e-3c52-4c4a-a36d-ba6800d29bd7
minimum(get_psr(chain_num_clusters_rj, 1000:10:size(chain_num_clusters_cg, 1)))

# ╔═╡ 65fbe8af-0c79-4710-a7a2-9b5b1b456a42
md"""
## DPM
"""

# ╔═╡ 77f2fb41-c0bc-4aac-a98d-0048c7b2a8b0
md"""
## Compare summary statistics

Plot the accuracy and the number of clusters.
"""

# ╔═╡ e6901d7d-6f3f-493e-9d00-7524475c5ccb
begin
	num_samples = 1000
end;

# ╔═╡ 3e429d3a-1d3e-48ff-b5aa-37e1bb0ec65c
begin

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
# ╟─31444176-7908-4eef-865d-4096aed328cd
# ╟─d98caa8b-0c20-4b41-b3e2-404061a6f575
# ╠═60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# ╟─5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
# ╠═f1e0e42f-c321-4969-b1cd-c0385fa73ae5
# ╠═6234628c-0274-4b23-9255-69e5f6878549
# ╠═db0fa5a1-11f8-44fa-a17a-814c9ffcfcf8
# ╠═058b8bb9-976a-4085-844c-1fa3fb5cb3a4
# ╟─0a91fe9a-1f4e-4b10-beef-896d41fcadd3
# ╠═270448bf-3eea-4ce6-87f5-f986d2e05057
# ╟─1401a242-3e07-4d2d-be8b-ec5599868457
# ╠═7560d033-9b51-4812-a857-19862b1767ec
# ╟─a274e4f7-b8d7-4c8a-a730-70889b0126ba
# ╠═f20d5e13-172c-4955-82d5-0248ab48cad4
# ╟─8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
# ╠═6044fecb-479f-49d2-ab76-f30cdbae0691
# ╠═b4180fdc-b209-414e-a028-b7890e69c302
# ╠═b301bb90-2178-4d49-bca2-e1f7ce59975f
# ╟─27a3553e-9211-45b8-b963-55c4511e6917
# ╟─d83f7bf8-552a-4c1f-ab09-6c394d2deb4e
# ╠═9a9d1185-0e58-40e5-a4b0-e5efb1f9d448
# ╠═d35f633b-5df3-4386-84fd-8b609462ac72
# ╠═ed090860-0352-4a63-9675-80a80e71cbeb
# ╠═b48c741d-b522-4314-8a95-ca5ff2089797
# ╠═db3947df-abe6-4349-ba56-3d89b72d5a1a
# ╠═4929b857-51bd-4c2d-9cf8-a0d59820f502
# ╠═d94cd58d-38a5-4b9e-be60-d252491af4ea
# ╠═84d56a3d-f364-4089-8808-289dd064e2f0
# ╠═6650bf54-e43a-4916-a0c7-1c0b0f0e2f4e
# ╟─ff2163c6-7cc8-47de-8acf-54295282eab6
# ╟─8f65e600-93c1-4a37-8a9e-9b187379689d
# ╠═1f0061bf-2699-4e6d-bcbe-2c5fb4287d7a
# ╠═4a008c9b-fbc4-407c-91cf-d5ca9e0a4662
# ╠═69e40e4e-3c52-4c4a-a36d-ba6800d29bd7
# ╟─65fbe8af-0c79-4710-a7a2-9b5b1b456a42
# ╟─77f2fb41-c0bc-4aac-a98d-0048c7b2a8b0
# ╠═e6901d7d-6f3f-493e-9d00-7524475c5ccb
# ╠═3e429d3a-1d3e-48ff-b5aa-37e1bb0ec65c
