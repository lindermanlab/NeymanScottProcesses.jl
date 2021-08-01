### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ fed40468-df48-11eb-13bc-95f2af985a19
begin
	using Pkg; Pkg.activate()
	using LinearAlgebra
	using Random
	using Distributions
	using Plots, LaTeXStrings
	using NeymanScottProcesses
	
	using Base.Iterators: product
	
	theme(:default, label=nothing, tickfont=(:Times, 8), guidefont=(:Times, 8))
end

# ╔═╡ 8d4761b5-8b0b-432f-8561-3224c9233946
md"
### Script parameters
"

# ╔═╡ dc9d79e6-bf85-48d3-b02b-e340ee7d6ac7
begin	
	# Model parameters
	model_dim = 2
	V = 1
	
	# Cluster parameters	
	max_clusters = 1000
	Ψ = 1e-3 * Matrix(I, model_dim, model_dim)
	ν = 5.0
	
	# Data parameters
	num_datapoints = 500
	
	β = 2.0
	αη = 1.0
	
	α_arr = [0.0, 1.0, 100.0, 10_000.0]  # NSP rate params
	γ_arr = [1.0, 10.0]  # DPM rate params
end

# ╔═╡ 7ba7277a-5a10-4556-8a76-ded6b050a48a
md"
### Sample clusters and potential datapoints
"

# ╔═╡ 68d4b439-61db-4b83-bcdc-ada1bb7f03dc
draw_cluster() = (μ=rand(model_dim), Σ=rand(InverseWishart(ν, Ψ)));

# ╔═╡ 1acf6f72-759a-4c5f-968e-c1fc42a51879
begin
	Random.seed!(512)
	clusters = [draw_cluster() for _ in 1:max_clusters]
end;

# ╔═╡ c6109d61-5a39-40dd-825b-db8da36fb18e
clusters[1]

# ╔═╡ cbfc618e-55cb-4b87-b26a-02b86ce70935
sample_potential_data(C) = [rand(MvNormal(C.μ, C.Σ)) for _ in 1:num_datapoints];

# ╔═╡ 36d98589-63f4-4e8e-af87-af3efe7d4197
begin
	Random.seed!(215)
	potential_data = [sample_potential_data(C) for C in clusters]
end;

# ╔═╡ ef211be5-b399-4653-855b-f92bc0526761
potential_data[3][5]

# ╔═╡ 9388f284-9134-4fa9-abb5-c7b926a54c6c
md"
### Sample datapoints using urn process
"

# ╔═╡ 0494486f-84db-41ee-9d13-dfb4560a25b1
begin
	Random.seed!(152)
	
	datasets = Dict()
	for (α, γ) in product(α_arr, γ_arr)
		data = []
		
		# The event rate varies according to concentration (γ) and cluster size (α)
		# η = γ / (V * α * (β / (1+β))^α)
		# or
		# γ = η * V * α * (β / (1+β))^α
		
		for _ in 1:num_datapoints
			
			existing_clusters = sort(unique(data))

			# Compute (unnormalized) probabilities
			probs = zeros(length(existing_clusters) + 1)
			for C in existing_clusters
				probs[C] = sum(==(C), data) + α
			end
			probs[end] = γ
			
			# Normalize
			probs = probs / sum(probs)
			
			# Sample
			push!(data, rand(Categorical(probs)))
		end
		
		datasets[(α, γ)] = data
	end
end

# ╔═╡ bf928bd7-351b-4a12-b9b5-619df91ffd96
md"
### Plot results
"

# ╔═╡ 4c622d22-496f-43c9-bda8-0853f6eddc55
begin
	plts = []
	for (i, j) in product(1:length(α_arr), 1:length(γ_arr))
		α, γ = α_arr[i], γ_arr[j] 
		assignments = datasets[(α, γ)]
		
		# Realize data
		realized_clusters = unique(assignments)
		cluster_counts = [count(==(C), assignments) for C in realized_clusters]
		samples = append!([
				potential_data[C][1:n_C] 
				for (C, n_C) in zip(realized_clusters, cluster_counts)
		]...)
		assignments = append!([
				fill(C, n_C) 
				for (C, n_C) in zip(realized_clusters, cluster_counts)
		]...)
	
		# Plot data
		plt = plot(xticks=nothing, yticks=nothing, aspect_ratio=:equal)
		plot!(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
		scatter!(
			[x[1] for x in samples], [x[2] for x in samples], 
			ms=4, msw=0.1, c=assignments
		)
		
		# Add labels if needed
		ylabel = (i == 1) ? L"\gamma = %$γ" : ""
		xlabel = (j == 2) ? L"\alpha = %$α" : ""
		plot!(ylabel=ylabel, xlabel=xlabel)
		plot!(frame=:box)
		
		push!(plts, plt)
	end
	main_plt = plot(plts..., layout=(length(γ_arr), length(α_arr)), size=(600, 300))
	plot!(bottom_margin=10Plots.pt, left_margin=20Plots.pt, bgoutside=:White)
	savefig(main_plt, "../../figures/example_data.pdf")
	savefig(main_plt, "../../figures/example_data.png")
	main_plt
end

# ╔═╡ 04817417-00bb-46e0-be1b-3dcc89f9f106


# ╔═╡ 913b02f5-f370-4743-aec3-3e211e02a122


# ╔═╡ 776c393d-9e01-4569-83aa-6a4f8b1a7744


# ╔═╡ 2164bf6a-27c6-447b-be9f-c7d3d45c3417


# ╔═╡ Cell order:
# ╠═fed40468-df48-11eb-13bc-95f2af985a19
# ╟─8d4761b5-8b0b-432f-8561-3224c9233946
# ╠═dc9d79e6-bf85-48d3-b02b-e340ee7d6ac7
# ╟─7ba7277a-5a10-4556-8a76-ded6b050a48a
# ╠═68d4b439-61db-4b83-bcdc-ada1bb7f03dc
# ╠═1acf6f72-759a-4c5f-968e-c1fc42a51879
# ╠═c6109d61-5a39-40dd-825b-db8da36fb18e
# ╠═cbfc618e-55cb-4b87-b26a-02b86ce70935
# ╠═36d98589-63f4-4e8e-af87-af3efe7d4197
# ╠═ef211be5-b399-4653-855b-f92bc0526761
# ╟─9388f284-9134-4fa9-abb5-c7b926a54c6c
# ╠═0494486f-84db-41ee-9d13-dfb4560a25b1
# ╟─bf928bd7-351b-4a12-b9b5-619df91ffd96
# ╠═4c622d22-496f-43c9-bda8-0853f6eddc55
# ╠═04817417-00bb-46e0-be1b-3dcc89f9f106
# ╠═913b02f5-f370-4743-aec3-3e211e02a122
# ╠═776c393d-9e01-4569-83aa-6a4f8b1a7744
# ╠═2164bf6a-27c6-447b-be9f-c7d3d45c3417
