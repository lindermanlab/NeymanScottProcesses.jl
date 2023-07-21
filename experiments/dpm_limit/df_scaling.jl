### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 2cb84200-b859-11ed-0020-23ee864f0ed2
using DrWatson

# ╔═╡ 6c777c22-2146-4b61-8cc6-04ba3e79f307
@quickactivate

# ╔═╡ 1139ab9b-0e48-43ff-a012-45f08c01081f
using Revise, NeymanScottProcesses

# ╔═╡ 2445cd91-373f-4891-97a5-fab4ee0f5434
using PlutoUI; TableOfContents()

# ╔═╡ 627eadca-87e1-449d-8df5-19a6f60fd0e1
using Random

# ╔═╡ 6c4603b7-ca61-4a04-ac66-2aa8b3552c74
using LinearAlgebra

# ╔═╡ 9b294339-2cd5-4651-846a-292d163e81e4
using MCMCDiagnosticTools

# ╔═╡ 1c7e8fa1-b739-4231-b950-191cc0bf009f
using Plots, StatsPlots

# ╔═╡ 0666f452-8a96-4e50-be06-8a04aff88020
theme(:dao; label=nothing, titlefontsize=10)

# ╔═╡ 69fa5848-466b-43e4-b56b-ede2afa28938
function generate_data(config)
	# Unpack config
	@unpack data_seed, cov_scale = config
	
	dim = get(config, :dim, 2) 
	
	η = get(config, :cluster_rate, 10.0)
	Ak = specify_gamma(10.0, 3.0^2)
	A0 = specify_gamma(0.1, 1.0^2)
	ν = get(config, :df, 5.0)

	bounds = Tuple(1.0 for _ in 1:dim)	
	
	# Set seed
	Random.seed!(data_seed)
	
	# Build priors
	Ψ = cov_scale * I(dim)  # Covariance scale
	priors = GaussianPriors(η, Ak, A0, Ψ, ν)
	
	# Build model
	gen_model = GaussianNeymanScottModel(bounds, priors)
	
	# Sample data
	data, assignments, clusters = sample(gen_model; resample_latents=true)
	data = Vector{RealObservation{dim}}(data)
	
	return @strdict(priors, gen_model, data, assignments, clusters)
end

# ╔═╡ e5b88d9c-a584-4249-9061-4573c98eecdb
function fit_data(config)
	# Unpack config
    @unpack data_seed, cov_scale, model_seed = config
	@unpack base_sampler_type, max_num_samples, max_time = config

	num_split_merge = get(config, :num_split_merge, 0) 
	split_merge_gibbs_moves = get(config, :split_merge_gibbs_moves, 0)
	num_jump_move = get(config, :num_jump_move, 10)

	# Load raw data
	dim = get(config, :dim, 2)
	df = get(config, :df, 5.0)
	bnds = Tuple(1.0 for _ in 1:dim)
	
	obs = generate_data(@dict(data_seed, cov_scale, dim, df))
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
			num_move=num_jump_move,
			split_merge_gibbs_moves=split_merge_gibbs_moves)
	else
		error("Invalid sampler type")
	end

	# Initialize model
 	model = GaussianNeymanScottModel(bnds, priors)
	z0 = Int.(rand(1:length(data), length(data)))

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

# ╔═╡ e4337846-8d53-4abc-a179-2217d23d581b
num_clusters(r::NamedTuple) = [
	length(unique(r.assignments[k][r.assignments[k] .!= -1])) 
	for k in 1:length(r.assignments)
]

# ╔═╡ f11d7e0e-d4f1-414e-9fa3-b4415bfff9fd
function make_chain(rs, f)
	data = [f(r) for r in rs]
	n = minimum(length.(data))

	num_chains = length(rs)
	
	return reshape(
		hcat([x[1:n] for x in data]...), 
		:, 1, num_chains
	)
end

# ╔═╡ 8cfedd15-d251-4236-8c4d-3906b2bfaee2
ess_method(chain) = ess_rhat(chain; method=ESSMethod())

# ╔═╡ 2e77970e-7b68-4902-b143-0eda47a914b8
get_ess(chain, samples) = [ess_method(chain[1:s, :, :])[1][1] for s in samples]

# ╔═╡ 45042524-4650-491d-9573-62da4f1fdfd7
function compute_ess(results, params, seeds; T=60, algs=["rj", "gibbs"])
	ess_results = Dict()
	
	for (a, d) in Iterators.product(algs, params)
		
		# Extract results
		r = [results[(d, s, a)] for s in seeds]

		# Make chains
		chain_t = make_chain(r, get_runtime)
		chain_K = make_chain(r, num_clusters)
	
		# Compute last sample in T seconds
		t = mean(chain_t, dims=[2, 3])[:]
		final_sample = findfirst(>(T), t)
	
		# Compute final effective sample size
		ess = last(get_ess(chain_K, 1:final_sample))
		@show a, d, ess

		ess_results[(a, d)] = ess
	end

	return ess_results
end

# ╔═╡ 53bf5daa-50ed-4957-9390-dc0049bbd59d
function make_data_plot(x, y)
	plt = plot(xticks=nothing, yticks=nothing, xlim=(0, 1), ylim=(0, 1),
	frame=:box)
	scatter!(x, y, c=:black, ms=1.5, alpha=0.5)
	return plt
end

# ╔═╡ ccb5c10b-9f76-403a-82c0-ec1270a51cf9
function plot_clusters!(plt, clusters; c=1)
	for C in clusters
		covellipse!(
			plt, C.sampled_position, C.sampled_covariance, 
			n_std=2, aspect_ratio=1, 
			alpha=0.3, c=c
		)
	end
end

# ╔═╡ d62edef6-0de3-4676-b0d4-2b6f9d534a22
md"""
## Debug
"""

# ╔═╡ 8be943be-e2b2-4762-90be-d8dd67772a39
iwmean(ψ, ν, d=2) = ψ / (ν - d - 1)

# ╔═╡ c3be91af-77c6-48a2-9259-f6a8bc635fe8
iwmean(0.001, 5.0)  # Main experiment

# ╔═╡ 9128dfbe-64ae-4344-9de9-05fc6e537429
iwmean(0.05^2 * 100.0, 100.0)  # Scaling experiment

# ╔═╡ 49f87549-22f6-4a43-9156-a7d36a356af9
iwmean(0.001 * 100.0, 100.0)  # Interpolate?

# ╔═╡ 777d6e25-8a8c-4007-93b8-a771953ce0de
function plot_data(; data_seed=1, cov_scale=0.001, df=5.0, kwargs...)
	observations = generate_data(Dict(
		:data_seed => data_seed, 
		:cov_scale => cov_scale,
		:df => df,
	))

	data = observations["data"]
	clusters = observations["clusters"]
	
	x = [p.position[1] for p in data]
	y = [p.position[2] for p in data]
	
	plt_true_data = make_data_plot(x, y)
	plot_clusters!(plt_true_data, clusters)	
	
	return plot!(; size=(200, 200), kwargs...)
end

# ╔═╡ efe74528-c4f6-47ac-a597-f416727c7dce
plot(
	plot_data(cov_scale=0.001, df=5.0, title="Main Figure"),
	plot_data(cov_scale=0.001 * 100, df=100.0, title="Dimension Experiment"),
	layout=(1, 3),
	size=(650, 200),
)

# ╔═╡ e7886936-8d6c-4b20-9b39-58514c23fece
md"""
## Generate Results
"""

# ╔═╡ e1d35b1c-a025-4d66-a3fd-a8070d233889
config = Dict(
	# Data
	:dim => 2,
	:data_seed => 1,
	:cov_scale => 0.001,
	:df => collect(5:5:100.0),

	# Model, required
	:model_seed => collect(1:3),
	:base_sampler_type => ["rj", "gibbs"],
	:max_num_samples => 10_000_000,
	:max_time => 2 * 60.0,
	:num_jump_move => 10,

	# Model, optional
	:num_split_merge => @onlyif(:base_sampler_type == "gibbs", 10),
	:split_merge_gibbs_moves => @onlyif(:num_split_merge > 0, 1),
)

# ╔═╡ bde76d4a-5f63-43ca-8904-3c119a1d5ff8
length(dict_list(config))

# ╔═╡ 2e480c82-d898-4111-9d04-cb69db0df57f
begin
	models = Dict()
	results = Dict()
	
	for c in dict_list(config)
		# Load key parameters for naming
		@unpack dim, model_seed, base_sampler_type = c
		
		# Run simulation		
		r, _ = produce_or_load(
			datadir("fit"),
			c,
			fit_data;
			force=false,
		)

		# Save result
		t = (dim, model_seed, base_sampler_type)
		models[t] = r["model"]
		results[t] = r["results"]
	end

	"Results Generated"
end

# ╔═╡ bbc70e14-bb44-4248-9236-4135cd0448d1
let
	N = 4
	s_max = 1
	
	plt = plot(
		size=(650, 250),
		ylims=(0, 30),
		xlims=(-1, 60)
	)

	for i in 1:s_max
		r = results[(N, i, "rj")]
		plot!(get_runtime(r), num_clusters(r), c=1, alpha=0.75)
	end

	for i in 1:s_max
		r = results[(N, i, "gibbs")]
		plot!(get_runtime(r), num_clusters(r), c=2, alpha=0.75)
	end

	plt
end

# ╔═╡ 679b1761-8dfc-4fd1-aa82-64d3d0e67fd2
ess = compute_ess(results, config[:dim], config[:model_seed])

# ╔═╡ f2a7afc1-1111-4166-bd0b-64bcd8361e21
md"""
## Plot Results
"""

# ╔═╡ 04c001a3-c087-45eb-9efa-5679199aeaf4
plt_dim = let

	dims = config[:dim]

	rj_ess = [ess[("rj", c)] for c in dims]
	cg_ess = [ess[("gibbs", c)] for c in dims]

	# Set up plot
	plt = plot(
		size=(650, 150), 
		legend=:outerright,
		xlabel="Dimension",
		bottom_margin=4Plots.mm,
		ylim=(-10, maximum(values(ess)) * 1.1),
	)

	# Plot each curve
	plot!(dims, cg_ess, lw=2, label="CG")
	plot!(dims, rj_ess, lw=2, label="RJ")

	plt
end

# ╔═╡ Cell order:
# ╠═2cb84200-b859-11ed-0020-23ee864f0ed2
# ╠═6c777c22-2146-4b61-8cc6-04ba3e79f307
# ╠═2445cd91-373f-4891-97a5-fab4ee0f5434
# ╠═1139ab9b-0e48-43ff-a012-45f08c01081f
# ╠═627eadca-87e1-449d-8df5-19a6f60fd0e1
# ╠═6c4603b7-ca61-4a04-ac66-2aa8b3552c74
# ╠═9b294339-2cd5-4651-846a-292d163e81e4
# ╠═1c7e8fa1-b739-4231-b950-191cc0bf009f
# ╠═0666f452-8a96-4e50-be06-8a04aff88020
# ╟─69fa5848-466b-43e4-b56b-ede2afa28938
# ╟─e5b88d9c-a584-4249-9061-4573c98eecdb
# ╟─e4337846-8d53-4abc-a179-2217d23d581b
# ╟─f11d7e0e-d4f1-414e-9fa3-b4415bfff9fd
# ╟─8cfedd15-d251-4236-8c4d-3906b2bfaee2
# ╟─2e77970e-7b68-4902-b143-0eda47a914b8
# ╟─45042524-4650-491d-9573-62da4f1fdfd7
# ╟─53bf5daa-50ed-4957-9390-dc0049bbd59d
# ╟─ccb5c10b-9f76-403a-82c0-ec1270a51cf9
# ╟─d62edef6-0de3-4676-b0d4-2b6f9d534a22
# ╠═8be943be-e2b2-4762-90be-d8dd67772a39
# ╠═c3be91af-77c6-48a2-9259-f6a8bc635fe8
# ╠═9128dfbe-64ae-4344-9de9-05fc6e537429
# ╠═49f87549-22f6-4a43-9156-a7d36a356af9
# ╟─777d6e25-8a8c-4007-93b8-a771953ce0de
# ╠═efe74528-c4f6-47ac-a597-f416727c7dce
# ╠═bbc70e14-bb44-4248-9236-4135cd0448d1
# ╟─e7886936-8d6c-4b20-9b39-58514c23fece
# ╠═e1d35b1c-a025-4d66-a3fd-a8070d233889
# ╠═bde76d4a-5f63-43ca-8904-3c119a1d5ff8
# ╠═2e480c82-d898-4111-9d04-cb69db0df57f
# ╠═679b1761-8dfc-4fd1-aa82-64d3d0e67fd2
# ╟─f2a7afc1-1111-4166-bd0b-64bcd8361e21
# ╠═04c001a3-c087-45eb-9efa-5679199aeaf4
