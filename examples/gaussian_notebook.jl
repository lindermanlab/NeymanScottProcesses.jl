### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 9a1d52aa-8e36-11ee-3f05-d14d87cc0fd2
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ 5548dd1c-01e3-4ac3-a6fd-c9abfd122bdd
using NeymanScottProcesses

# ╔═╡ 4eb68bfc-f91d-4894-9cc1-8e2721fe313c
using Plots

# ╔═╡ dd661d0f-34e1-454f-9d88-3cf69821138e
using LinearAlgebra: I

# ╔═╡ 464656c3-ab06-4b25-acdd-fcb8c5db0adf
using Random: seed!

# ╔═╡ 0be04ce6-b884-4a12-868f-ce991df43b98
using PlutoUI; TableOfContents()

# ╔═╡ 660b4204-e9ff-41d3-98ee-08cccf7d34a1
seed!(1234)

# ╔═╡ 3c058b9e-6456-4c97-9481-a2567623ab96
md"""
# Parameters
"""

# ╔═╡ 63faf5ed-3585-43e3-bedd-3e90376375a4
begin
	dim = 2  # Dimension of the data
	bounds = Tuple(4.0 for _ in 1:dim)  # Model bounds
	max_cluster_radius = 0.5
	
	K = 4.0  # Cluster rate
	Ak = specify_gamma(20.0, 3.0)  # Cluster amplitude
	A0 = specify_gamma(20.0, 3.0)  # Background amplitude
	
	Ψ = 1e-3 * I(dim)  # Covariance scale
	ν = 5.0  # Covariance degrees of freedom
	
	mask_radius = 0.05
	percent_masked = 0.20
end;

# ╔═╡ 93ee5b46-ca92-4eda-9d73-90d73499fb71
md"""
# Generative Model
"""

# ╔═╡ 57b7f704-0e47-4c08-bc45-50e8a261965a
gen_priors = GaussianPriors(K, Ak, A0, Ψ, ν)

# ╔═╡ 2da3804e-2310-4315-8735-8b462b732844
gen_model = GaussianNeymanScottModel(bounds, gen_priors)

# ╔═╡ 66c2da10-31b0-4067-89ed-ac1cc665acae
data, assignments, clusters = sample(gen_model; resample_latents=true)

# ╔═╡ 13a4f8e5-02e5-430f-a283-0a354b288afd
md"""
### Generate masks
"""

# ╔═╡ d35ac347-f9ef-4df6-8e17-d3c7051c853b
masks = create_random_mask(gen_model, mask_radius, percent_masked);

# ╔═╡ 4208a33c-c4b1-48e4-b1bb-8a0066ed93cf
masked_data, unmasked_data = split_data_by_mask(data, masks);

# ╔═╡ 03d81a4b-96b3-4056-8709-2c5b44b724f1
md"""
### Visualize data
"""

# ╔═╡ b9a633cc-a658-4043-bc76-222b91486836
begin
	p1 = plot(data, assignments, xlim=(0, 2), ylim=(0, 2), size=(300, 300), title="")
	plot!(p1, masks)
	
	@show length(data)
	p1
end

# ╔═╡ 720d1fe5-ad5a-4bbb-9d69-448a04c5f21e
md"""
# Inference
"""

# ╔═╡ 0317c2cd-d8d0-495e-b591-126f46685be3
md"""
### Construct sampler
"""

# ╔═╡ c425f4e4-b53b-48fb-9a57-26277bfb26c8
begin
	base_sampler = GibbsSampler(num_samples=50, save_interval=10)
	
	masked_sampler = MaskedSampler(base_sampler, masks; masked_data=masked_data, num_samples=3)
	
	sampler = Annealer(masked_sampler, 200.0, :cluster_amplitude_var; num_samples=3)
end

# ╔═╡ d8a77ac9-a1fd-4e99-9446-2ff565353b0f
md"""
### Create model and sample
"""

# ╔═╡ 9a225e8f-fe53-45cc-bfe2-fecdca3139e7
priors = deepcopy(gen_priors)

# ╔═╡ 20671b1a-1966-4b92-8f1f-42c961c0a93c
md"""
### Run sampler
"""

# ╔═╡ 9acbf8ad-5c69-4e86-a0b8-a1ec5e2520b8
begin
	model = GaussianNeymanScottModel(bounds, priors; max_radius=max_cluster_radius)
	results = sampler(model, unmasked_data)
end

# ╔═╡ 56efc04d-368f-4768-a6de-3a5f84f99b12
sampled_data, sampled_assignments = sample_masked_data(model, masks)

# ╔═╡ b7378cee-efe6-42eb-8e57-9484d317aff2
begin
	# Visualize results
	p2 = plot(
	    unmasked_data, last(results.assignments);
	    size=(400, 400), xlim=(0, 2), ylim=(0, 2), title="estimate"
	)
	plot!(p2, masks)
	plot!(p2, sampled_data, color="red")
	
	plot(p1, p2, layout=(1, 2), size=(800, 400))
end

# ╔═╡ Cell order:
# ╠═9a1d52aa-8e36-11ee-3f05-d14d87cc0fd2
# ╠═5548dd1c-01e3-4ac3-a6fd-c9abfd122bdd
# ╠═4eb68bfc-f91d-4894-9cc1-8e2721fe313c
# ╠═dd661d0f-34e1-454f-9d88-3cf69821138e
# ╠═464656c3-ab06-4b25-acdd-fcb8c5db0adf
# ╠═660b4204-e9ff-41d3-98ee-08cccf7d34a1
# ╠═0be04ce6-b884-4a12-868f-ce991df43b98
# ╟─3c058b9e-6456-4c97-9481-a2567623ab96
# ╠═63faf5ed-3585-43e3-bedd-3e90376375a4
# ╟─93ee5b46-ca92-4eda-9d73-90d73499fb71
# ╠═57b7f704-0e47-4c08-bc45-50e8a261965a
# ╠═2da3804e-2310-4315-8735-8b462b732844
# ╠═66c2da10-31b0-4067-89ed-ac1cc665acae
# ╟─13a4f8e5-02e5-430f-a283-0a354b288afd
# ╠═d35ac347-f9ef-4df6-8e17-d3c7051c853b
# ╠═4208a33c-c4b1-48e4-b1bb-8a0066ed93cf
# ╟─03d81a4b-96b3-4056-8709-2c5b44b724f1
# ╠═b9a633cc-a658-4043-bc76-222b91486836
# ╟─720d1fe5-ad5a-4bbb-9d69-448a04c5f21e
# ╟─0317c2cd-d8d0-495e-b591-126f46685be3
# ╠═c425f4e4-b53b-48fb-9a57-26277bfb26c8
# ╟─d8a77ac9-a1fd-4e99-9446-2ff565353b0f
# ╠═9a225e8f-fe53-45cc-bfe2-fecdca3139e7
# ╟─20671b1a-1966-4b92-8f1f-42c961c0a93c
# ╠═9acbf8ad-5c69-4e86-a0b8-a1ec5e2520b8
# ╠═56efc04d-368f-4768-a6de-3a5f84f99b12
# ╠═b7378cee-efe6-42eb-8e57-9484d317aff2
