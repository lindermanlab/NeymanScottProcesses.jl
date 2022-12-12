### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ dd0f7868-4a6c-4def-b17c-1f7835cba864
using DrWatson

# ╔═╡ 98d77a92-3744-4df3-a367-df22f3612e9d
@quickactivate

# ╔═╡ acc93e48-e012-11eb-048e-35c010d6acee
begin
	using Plots
	using Random, StatsPlots
	
	using Revise, NeymanScottProcesses
	
	using LinearAlgebra: I
	using Random: seed!
end

# ╔═╡ 9a9d1185-0e58-40e5-a4b0-e5efb1f9d448
using MCMCDiagnosticTools

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
## Fixed Parameters
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
	ν = 5.0  # Covariance degrees of freedom
end;

# ╔═╡ b9ee61fa-c387-404b-b273-11dcfa8b63a0
md"""
## Generate data
"""

# ╔═╡ fbe2709e-adaf-40a3-bb40-ed4243ff62cd
function generate_data(config)
	# Unpack config
    @unpack data_seed, cov_scale = config

	# Set seed
    Random.seed!(data_seed)

	# Build priors
	Ψ = cov_scale * I(dim)  # Covariance scale
	priors = GaussianPriors(η, Ak, A0, Ψ, ν)

	# Build model
    gen_model = GaussianNeymanScottModel(bounds, priors)

	# Sample data
    data, assignments, clusters = sample(gen_model; resample_latents=true)
    data = Vector{RealObservation{2}}(data)

    return @strdict(priors, gen_model, data, assignments, clusters)
end

# ╔═╡ ccc338f5-baae-4009-ac9b-413b8b4aafbb
# Data parameters

# ╔═╡ d98caa8b-0c20-4b41-b3e2-404061a6f575
md"""
## Fit NSP and Gibbs and RJMC
"""

# ╔═╡ 990a0e9f-2bb5-4cac-8de9-7fd68500a53e
function fit_data(config)
	# Unpack config
    @unpack data_seed, cov_scale, model_seed = config
	@unpack base_sampler_type, max_num_samples, max_time = config

	num_split_merge = get(config, :num_split_merge, 0) 
	split_merge_gibbs_moves = get(config, :split_merge_gibbs_moves, 0)

	# Load raw data
	obs, _ = produce_or_load(
		datadir("observations"), 
		@dict(data_seed, cov_scale),
		generate_data)
	@unpack priors, gen_model, data, clusters, assignments = obs

	# Set model / chain seed
    Random.seed!(model_seed)

	# Choose base sampler
	if base_sampler_type == "gibbs"
		sampler = GibbsSampler(
			num_samples=max_num_samples,
			max_time=max_time,
			num_split_merge=num_split_merge, 
			save_interval=1,
			split_merge_gibbs_moves=split_merge_gibbs_moves,
			verbose=false)
	elseif base_sampler_type == "rj"
		sampler = ReversibleJumpSampler(
			num_samples=max_num_samples, 
			max_time=max_time,
			birth_prob=0.5,
			num_split_merge=num_split_merge, 
			split_merge_gibbs_moves=split_merge_gibbs_moves)
	else
		error("Invalid sampler type")
	end

	# Initialize model
 	model = GaussianNeymanScottModel(bounds, priors)
	z0 = rand(1:length(data), length(data))

	# Fit model
	t = @elapsed results = sampler(
		model, 
		data,
		initial_assignments=z0
	)
	avg_ll = NeymanScottProcesses.log_like(model, data) / length(data)

	println("Fit model in $t seconds")
	println("Average log likelihood: $(avg_ll)")
	
	return @strdict(model, results)
end

# ╔═╡ d05b5436-7713-48a3-b3f7-c23ecf1d3c8b
base_config = Dict(
	# Required
	:data_seed => 1,
	:cov_scale => 1e-3,
	:model_seed => 1, #[1, 2, 3],
	:base_sampler_type => "rj", #["rj", "gibbs"],
	:max_num_samples => 100_000,
	:max_time => 240.0, #120.0,

	# Optional
	:num_split_merge => [0, @onlyif(:base_sampler_type == "gibbs", 10)],
	:split_merge_gibbs_moves => [0, @onlyif(:num_split_merge > 0, 1)],
)

# ╔═╡ 8a9d96cb-3baf-433c-a8d4-4d8b83a753fd
data_seed = base_config[:data_seed]

# ╔═╡ ac60e75a-5691-469c-966b-fca70c2ee7fe
cov_scale = base_config[:cov_scale]

# ╔═╡ 010719a8-877c-4701-a8b6-3f56111fd735
observation_data, _ = produce_or_load(
	datadir("observations"),
	@dict(data_seed, cov_scale),
	generate_data
);

# ╔═╡ b100f88d-5158-4099-a7d9-6e9d433c2f14
@unpack priors, gen_model, data, clusters, assignments = observation_data;

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

# ╔═╡ a0db0704-8e60-4ff5-b300-1eae870b433c
dict_list(base_config)

# ╔═╡ 939cb916-9067-4ba7-86dd-c734a933c6e0
md"""
### NOTE: MAKE SURE FORCE IS FALSE
"""

# ╔═╡ 6ee153f0-b83f-4f1f-8b09-8cf4ef023ba9
begin
	models = Dict()
	results = Dict()
	
	for c in dict_list(base_config)
		# Load key parameters for naming
		@unpack model_seed, base_sampler_type = c
		@unpack num_split_merge, split_merge_gibbs_moves = c
		@show base_sampler_type, num_split_merge, model_seed, split_merge_gibbs_moves
		
		# Run simulation		
		r, _ = produce_or_load(
			datadir("fit"),
			c,
			fit_data;
			force=false,
		)

		# Save result
		t = (base_sampler_type, num_split_merge, split_merge_gibbs_moves, model_seed)
		models[t] = r["model"]
		results[t] = r["results"]
		@show last(get_runtime(r["results"]))
	end
end

# ╔═╡ 26f8ab35-6939-4017-9292-2d9fc8a558dd
r_gibbs = [results[("gibbs", 10, 1, c)] for c in 1:3]

# ╔═╡ f9048b2e-e744-4d49-905b-47ada5a84e54
r_gibbs_sm0 = [results[("gibbs", 10, 0, c)] for c in 1:3]

# ╔═╡ 098e22dc-223d-48d4-a865-bc6dfbedc39d
r_rj = [results[("rj", 0, 0, c)] for c in 1:3]

# ╔═╡ 73067b2c-df07-415f-9ccc-74c9e70dda64
model_gibbs = [models[("gibbs", 10, 1, c)] for c in 1:3]

# ╔═╡ 5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
md"""
## Plot results
"""

# ╔═╡ 058b8bb9-976a-4085-844c-1fa3fb5cb3a4
true_num_clusters = length(unique(assignments[assignments .!= -1]))

# ╔═╡ 6234628c-0274-4b23-9255-69e5f6878549
num_clusters(r::NamedTuple) = [length(unique(r.assignments[k][r.assignments[k] .!= -1])) for k in 1:length(r.assignments)]

# ╔═╡ db0fa5a1-11f8-44fa-a17a-814c9ffcfcf8
bkgd_rate(r::NamedTuple) = [r.globals[k].bkgd_rate for k in 1:length(r.globals)]

# ╔═╡ 270448bf-3eea-4ce6-87f5-f986d2e05057
let
	plt = plot(size=(400, 200), ylim=(0, 2*true_num_clusters), title="Number of clusters during CG algorithm", xlabel="Sample", xlim=(0, 1000))
	[plot!(num_clusters(r_gibbs[k]), lw=2) for k in 1:3]
	[plot!(num_clusters(r_gibbs_sm0[k]), lw=1, c=:gray) for k in 1:3]
	hline!([true_num_clusters], c=:Black, lw=2, label="True")
	plt
end

# ╔═╡ f20d5e13-172c-4955-82d5-0248ab48cad4
get_num_clusters(r) = num_clusters(r)

# ╔═╡ 8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
plt_cgbd_ll = let
	plot(size=(250, 200), dpi=200)
	
	plot!(
		[append!([0.0], r.log_p) for r in r_gibbs] / length(data), 
		lw=2, 
		c=1, 
		label=["CG" nothing nothing], 
		alpha=0.7
	)
	plot!(
		[append!([0.0], r.log_p) for r in r_rj] / length(data), 
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
		xlim=(-100, 3000),
		xticks=(0:1000:15_000),
		grid=false
	)
end

# ╔═╡ 6044fecb-479f-49d2-ab76-f30cdbae0691
plt_cgbd_nc = let
	plot(size=(250, 200), dpi=200)
	# plot!(
	# 	[get_runtime(r) for r in r_gibbs_sm0],
	# 	[get_num_clusters(r) for r in r_gibbs_sm0], 
	# 	lw=2, c=1, label=["CG SM-Random" nothing nothing]
	# )
	plot!(
		[get_runtime(r) for r in r_gibbs],
		[get_num_clusters(r) for r in r_gibbs], 
		lw=2, c=2, label=["CG" nothing nothing]
	)
	plot!(
		[get_runtime(r) for r in r_rj],
		[get_num_clusters(r) for r in r_rj], 
		lw=2, c=3, label=["RJ" nothing nothing], alpha=0.5
	)
	hline!([true_num_clusters], c=:Black, lw=2, label="True")
	
	
	plot!(
		ylabel="Number of Clusters", xlabel="Seconds", legend=:topright,
		ylim=(0, true_num_clusters*3), grid=false,
	)
end

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
	plot_clusters!(plt_fit_data_nsp, model_gibbs[1].clusters)
	plot!(plt_fit_data_nsp, title="Learned (NSP)")
	
	plot(plt_true_data, plt_fit_data_nsp, size=(500, 200))
	"Started NSP plot"
end

# ╔═╡ d83f7bf8-552a-4c1f-ab09-6c394d2deb4e
md"""
## MCMC Diagnostics
"""

# ╔═╡ ed090860-0352-4a63-9675-80a80e71cbeb
function make_chain(rs, f)
	data = [f(r) for r in rs]
	n = minimum(length.(data))
	
	return reshape(
		hcat([x[1:n] for x in data]...), 
		:, 1, num_chains
	)
end

# ╔═╡ b48c741d-b522-4314-8a95-ca5ff2089797
chain_num_clusters_cg = make_chain(r_gibbs, num_clusters);

# ╔═╡ db3947df-abe6-4349-ba56-3d89b72d5a1a
chain_num_clusters_rj = make_chain(r_rj, num_clusters);

# ╔═╡ 4929b857-51bd-4c2d-9cf8-a0d59820f502
size(chain_num_clusters_cg)

# ╔═╡ d94cd58d-38a5-4b9e-be60-d252491af4ea
ess_method(chain) = ess_rhat(chain; method=ESSMethod())

# ╔═╡ 84d56a3d-f364-4089-8808-289dd064e2f0
get_ess(chain, samples) = [ess_method(chain[1:s, :, :])[1][1] for s in samples]

# ╔═╡ 6650bf54-e43a-4916-a0c7-1c0b0f0e2f4e
get_psr(chain, samples) = [gelmandiag(chain[1:s, :, :])[2][1] for s in samples]

# ╔═╡ 1f0061bf-2699-4e6d-bcbe-2c5fb4287d7a
let
	_x1 = 50:10:size(chain_num_clusters_cg, 1)
	_x2 = 50:10:size(chain_num_clusters_rj, 1)

	t_cg = mean(make_chain(r_gibbs, get_runtime), dims=[2, 3])[:]
	t_rj = mean(make_chain(r_rj, get_runtime), dims=[2, 3])[:]
	
	plt1 = plot(title="Effective Sample Size", legend=:topleft)

	plot!(t_cg[_x1], get_ess(chain_num_clusters_cg, _x1), lw=3, label="CG + 10 split merge")
	plot!(t_rj[_x2], get_ess(chain_num_clusters_rj, _x2), lw=3, label="RJ (no split merge)")

	plt2 = plot(title="Potential Scale Reduction", ylim=(0.9, 1.1))
	hline!([1], color=:black, lw=2)
	plot!(t_cg[_x1], get_psr(chain_num_clusters_cg, _x1), lw=3, c=1)
	plot!(t_rj[_x2], get_psr(chain_num_clusters_rj, _x2), lw=3, c=2)

	plot(plt1, plt2, layout=(2, 1), xlim=(0, 100))
	
end

# ╔═╡ Cell order:
# ╠═dd0f7868-4a6c-4def-b17c-1f7835cba864
# ╠═98d77a92-3744-4df3-a367-df22f3612e9d
# ╠═acc93e48-e012-11eb-048e-35c010d6acee
# ╠═5de272b0-931a-4851-86ac-4249860a9922
# ╟─be352c60-88b9-421a-8fd7-7d34e19665e6
# ╟─d1d53b74-3e7b-44eb-b0e7-1e4b612edeb2
# ╠═b0114a0b-58e4-44d2-82db-3a6131435b32
# ╟─b9ee61fa-c387-404b-b273-11dcfa8b63a0
# ╟─fbe2709e-adaf-40a3-bb40-ed4243ff62cd
# ╠═ccc338f5-baae-4009-ac9b-413b8b4aafbb
# ╠═8a9d96cb-3baf-433c-a8d4-4d8b83a753fd
# ╠═ac60e75a-5691-469c-966b-fca70c2ee7fe
# ╠═010719a8-877c-4701-a8b6-3f56111fd735
# ╠═b100f88d-5158-4099-a7d9-6e9d433c2f14
# ╠═10924047-b95e-41db-86ef-6f5c2cbea1a5
# ╠═31444176-7908-4eef-865d-4096aed328cd
# ╟─d98caa8b-0c20-4b41-b3e2-404061a6f575
# ╟─990a0e9f-2bb5-4cac-8de9-7fd68500a53e
# ╠═d05b5436-7713-48a3-b3f7-c23ecf1d3c8b
# ╠═a0db0704-8e60-4ff5-b300-1eae870b433c
# ╟─939cb916-9067-4ba7-86dd-c734a933c6e0
# ╠═6ee153f0-b83f-4f1f-8b09-8cf4ef023ba9
# ╠═26f8ab35-6939-4017-9292-2d9fc8a558dd
# ╠═f9048b2e-e744-4d49-905b-47ada5a84e54
# ╠═098e22dc-223d-48d4-a865-bc6dfbedc39d
# ╠═73067b2c-df07-415f-9ccc-74c9e70dda64
# ╟─5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
# ╠═058b8bb9-976a-4085-844c-1fa3fb5cb3a4
# ╠═6234628c-0274-4b23-9255-69e5f6878549
# ╠═db0fa5a1-11f8-44fa-a17a-814c9ffcfcf8
# ╠═270448bf-3eea-4ce6-87f5-f986d2e05057
# ╠═f20d5e13-172c-4955-82d5-0248ab48cad4
# ╟─8b5b0efa-8be5-4b29-8531-1f52abb8ebf7
# ╟─6044fecb-479f-49d2-ab76-f30cdbae0691
# ╠═b301bb90-2178-4d49-bca2-e1f7ce59975f
# ╟─27a3553e-9211-45b8-b963-55c4511e6917
# ╟─d83f7bf8-552a-4c1f-ab09-6c394d2deb4e
# ╠═9a9d1185-0e58-40e5-a4b0-e5efb1f9d448
# ╠═ed090860-0352-4a63-9675-80a80e71cbeb
# ╠═b48c741d-b522-4314-8a95-ca5ff2089797
# ╠═db3947df-abe6-4349-ba56-3d89b72d5a1a
# ╠═4929b857-51bd-4c2d-9cf8-a0d59820f502
# ╠═d94cd58d-38a5-4b9e-be60-d252491af4ea
# ╠═84d56a3d-f364-4089-8808-289dd064e2f0
# ╠═6650bf54-e43a-4916-a0c7-1c0b0f0e2f4e
# ╠═1f0061bf-2699-4e6d-bcbe-2c5fb4287d7a
