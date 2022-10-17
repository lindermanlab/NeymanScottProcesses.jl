### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ c89df6e8-4338-11ec-2338-dd887304f326
using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# ╔═╡ 086186c4-55e5-46cc-9464-b5c81cde470e
using DataFrames, Dates, Distributions, Random, SparseArrays

# ╔═╡ 8ae7c137-d694-417d-9f93-33484766e1fb
using CSV, JLD

# ╔═╡ 41cea1a0-1ac5-4f99-a2df-bc82165f5ebc
using NeymanScottProcesses

# ╔═╡ 56b34d1d-81f9-4434-98c2-6185db8fe19a
using Plots

# ╔═╡ 4d98b0bd-3e93-4fff-9873-8c6474f38ed6
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 3de8b9b4-d60a-47a7-a084-97fde9de4feb
util = ingredients("util.jl")

# ╔═╡ c21040c8-269a-4239-a4f5-fe2c87074031
md"""
## Set up plotting
"""

# ╔═╡ 94fe7f2e-d82d-4929-ac3f-2b3e7784ce4f
theme(:default, label=nothing, grid=false, lw=2)

# ╔═╡ 03c03483-c87d-483e-82fe-721099bd89d3
md"""
## Load data
"""

# ╔═╡ c07b33b8-4c65-4da9-99a1-83fd8d499624
cfg = util.CONFIG

# ╔═╡ 0fc0e5ee-e631-4c7a-a762-a3d53c4a4268
max_time = (cfg[:max_date] - cfg[:min_date]).value

# ╔═╡ bb7af94b-5779-4f52-bef5-a90169c5fd14
base = JLD.load(joinpath(util.RESULTSDIR, "baseline_fit.jld"));

# ╔═╡ b69306f8-6e64-43b0-a4ce-7a4f3c9c2536
nsp = JLD.load(joinpath(util.RESULTSDIR, "nsp_fit.jld"));

# ╔═╡ 8f24232e-a7ea-4d87-8a64-9ef23fcac0d6
md"""
## Statistic: log likelihoods
"""

# ╔═╡ 3650e87b-2672-4398-9f04-8240194166d1


# ╔═╡ 65d6ece6-61ca-4c9c-a882-fbb3064e54f7
md"""
## Figure 1A: cluster intensities
"""

# ╔═╡ 6c37769b-f756-4e00-8762-dd669bcb1478
md"""
### Baseline
"""

# ╔═╡ f2afe3b5-ed3a-468d-a032-ede27ece64bd
base_amplitudes = base["model"].A[:, 1]

# ╔═╡ f8bf84c1-6372-4456-a6f7-ca6af03c91ed
num_bins = length(base_amplitudes)

# ╔═╡ eedf8b14-4e93-4832-85e6-5b2c04cfc385
bin_size = max_time / num_bins

# ╔═╡ 6173a155-f8ff-421d-be52-66c14b349338
get_timebin(t) = max(1, ceil(Int, t / bin_size))

# ╔═╡ 8c045873-bba5-4842-bbe6-2b6b466a1b53
t = 0 : 0.01 : (max_time - 1)

# ╔═╡ 01157707-9ff9-4610-8131-0471d9a44a0a
md"""
### Neyman-Scott
"""

# ╔═╡ 0379e63e-7e7a-40be-b319-3224fd1ec12d
nsp_clusters = nsp["model"].clusters.clusters

# ╔═╡ 78e483ff-058b-48c0-9da7-2def7e74dbde
cluster_intensity(C) = t -> C.sampled_amplitude * pdf(Normal(C.sampled_position, sqrt(C.sampled_variance)), t)

# ╔═╡ 6fca003b-d0f9-4d13-873c-68f4c74add38
λ = cluster_intensity.(nsp_clusters)

# ╔═╡ f1a2057e-2cb6-47d8-bd6f-a626af7d0321
num_clusters = length(λ)

# ╔═╡ 69edfdf0-077a-442b-91d5-40d51774d943
LINEWIDTH = 3

# ╔═╡ 3768c42c-8deb-4fc1-a76b-44e63f533d17
function plot_base_intensity(k_highlighted; kwargs...)	
	plot()
	for k in push!(collect(1:num_bins), k_highlighted)
		Ak = (get_timebin.(t) .== k)
		color = (k == k_highlighted) ? :blue : :lightgray
		
		plot!(t, Ak .* base_amplitudes[k], c=color, lw=LINEWIDTH)
	end
	
	return plot!(
		ylim=(0.1, maximum(base_amplitudes)+1000), 
		xlim=(minimum(t), maximum(t)),
		frame=:box,
		ticks=nothing,
		kwargs...
	)
end

# ╔═╡ d9d7da1d-8c29-4473-94e2-a83b60b10db1
base_intensity_plots = plot_base_intensity.(1:num_bins)

# ╔═╡ 19fa4c42-d92a-4e65-86cf-f68642c1eb5a
function plot_nsp_intensity(k_highlighted; kwargs...)
	λmax = 100 + maximum([maximum(λi.(t)) for λi in λ])
	
	
 	plot()
 	for λi in λ
 		plot!(t, λi.(t), lw=LINEWIDTH, c=:lightgray)
 	end
	plot!(t, λ[k_highlighted].(t), lw=LINEWIDTH, c=:blue)
	
	return plot!(
		ylim=(0.1, λmax), 
		xlim=(minimum(t), maximum(t)),
		frame=:box,
		ticks=nothing,
		kwargs...
	)
end

# ╔═╡ e54f802e-fda9-4843-8321-a4a1fdc52d3c
nsp_intensity_plots = plot_nsp_intensity.(1:num_clusters)

# ╔═╡ 3e054434-e7ce-4e95-aea7-0229ba192dd7
md"""
## Figure 1B: Cluster word distributions
"""

# ╔═╡ 585f7e78-e31e-4fed-8dc4-768070cf959a
data, Dω, meta, _, _, _ = util.load_results(cfg, "nsp_fit.jld");

# ╔═╡ 339ff2db-2757-4d20-8e5d-b6dded9954cc
begin
	Dϵ = zeros(size(Dω, 2))
	for x in data
		Dϵ[x.embassy] += 1
	end
end

# ╔═╡ 873815b6-9af4-4481-a634-a63d3d755429
begin
	function get_top_k_words(ω, k; min_length=3, max_length=20)
		n, m = size(Dω)

		# Other normalizations: Dω*ϵ .+ 1/n, ones(n)
		𝔼ω = Dω * ones(m) .+ sum(Dω) / n
		ω_scores = ω ./ 𝔼ω
		
		word_order = meta.vocab[sortperm(ω_scores, rev=true), :word]

		return filter(w -> max_length >= length(w) >= min_length, word_order)[1:k]
	end
	
	function nsp_get_top_k_words(i, k)
		return get_top_k_words(nsp_clusters[i].sampled_word_probs, k)
	end
	
	function baseline_get_top_k_words(τ, k)
		return get_top_k_words(base["model"].ω[:, τ, 1], k)
	end
end

# ╔═╡ b6772648-34fa-42f8-8fe5-ec62265950b9
begin
	function get_top_k_embassies(ϵ, k; min_count=10)
		embassy_order = sortperm(ϵ, rev=true)
		embassy_order = filter(i -> Dϵ[i] > min_count, embassy_order)
		return meta.embassies[embassy_order[1:k], :embassy]
	end
	
	function nsp_get_top_k_embassies(i, k)
		return get_top_k_embassies(nsp_clusters[i].sampled_embassy_probs, k)
	end
	
	function baseline_get_top_k_embassies(τ, k)
		return get_top_k_embassies(base["model"].ϵ[:, τ, 1], k)
	end
end

# ╔═╡ bfcee336-a4d7-43a2-904e-b24cc7f161dc
nsp_get_top_k_words(1, 30)

# ╔═╡ 13d1fc9e-fff3-40ff-a6ba-b5d60d48016e
nsp_get_top_k_embassies(1, 15)

# ╔═╡ bb743a40-e9a0-4c22-a54e-44dc4631afb0
findall(x -> x.embassy == "STATE", eachrow(meta.embassies))

# ╔═╡ e6721d4f-d746-4885-9a25-28995d938efa
Dϵ[184]

# ╔═╡ af716196-b8ec-4f8a-bd81-fb78b61be721
ENTEBBE_WORDS = Set(["ENTEBBE", "UGANDA", "HIJACKING", "PLO", "HOSTAGES"])

# ╔═╡ e6445104-e533-4967-bca8-c5674aa3f490
BICEN_WORDS = Set(["BICENTENNIAL", "TH ANNIVERSARY", "CONGRATULATIONS", "CENTURY", "AMERICAN PEOPLE"])

# ╔═╡ 504f4121-c295-4cb4-9b90-5545533a09d9
BOLD_WORDS = union(ENTEBBE_WORDS, BICEN_WORDS)

# ╔═╡ d876cfc9-aba8-4372-a9a2-1c435e1a7257
function plot_words(words)
	plot()

	for (i, w) in enumerate(words)
		font = w in BOLD_WORDS ? "times bold" : "times"
		
		color = :Black
		w in ENTEBBE_WORDS && (color = :Red)
		w in BICEN_WORDS && (color = :ForestGreen)
		
		annotate!(0.05, 1.04- 0.065*i, text(w, font, 6, :left, color, :top))
	end
	#words = text(join(words, "\n"), 6, "times", :left)
	#annotate!(0.05, 0.5, words)
	return plot!(
		frame=:box, 
		xticks=nothing, yticks=nothing, 
		bgcolour_inside=:WhiteSmoke,  
		size=(200, 0.8*200)
	)
end

# ╔═╡ de03c4a3-ec58-46b3-9c34-12f729d0f89c
nsp_word_plots = [plot_words(nsp_get_top_k_words(i, 15)) for i in 1:num_clusters]

# ╔═╡ 31e14f3c-b2c6-4d5a-b340-087d7c139a06
base_word_plots = [plot_words(baseline_get_top_k_words(i, 15)) for i in 1:num_bins]

# ╔═╡ 7a645f38-8046-4a2a-b075-dedd6bcfdaf3
md"""
## Figure 1C: Put it all together
"""

# ╔═╡ 4637b901-c112-4d13-91c9-0810906cfdd5
sortperm([C.sampled_amplitude for C in nsp_clusters], rev=true)
# Entebbe = 2, bicentennial = 1

# ╔═╡ 4a41a034-1822-41fb-8f5a-b450aa649479
sortperm([C.sampled_position for C in nsp_clusters])

# ╔═╡ e25fe7e2-a574-4623-b36b-6c9b79ca6d9a
best_nsp = [4; 1; 2; 3]

# ╔═╡ db5723a1-3077-4073-8913-ed62aeca520b
plt_nsp = let
	amplitudes = plot(nsp_intensity_plots[best_nsp]..., layout=(1, 4))
	words = plot(nsp_word_plots[best_nsp]..., layout=(1, 4))
	
	l = @layout [a{0.1h}; b]
	plot(amplitudes, words, layout=l, size=(600, 225))
end

# ╔═╡ e52847bd-81c1-4b61-971c-223929ed3bea
plt_base = let
	amplitudes = plot(base_intensity_plots..., layout=(1, num_bins))
	words = plot(base_word_plots..., layout=(1, num_bins))
	
	l = @layout [a{0.1h}; b]
	plot(amplitudes, words, layout=l, size=(600, 225))
end

# ╔═╡ f8bf18fb-f444-4234-a0bc-471318391879
plt_docs = plot(plt_nsp, plt_base, size=(600, 450), layout=(2, 1))

# ╔═╡ ce1f0a38-e8cb-45b4-bc15-c9b2c5a06afd
savefig(plt_docs, "../../figures/document_model.pdf")

# ╔═╡ f7f93a79-1b7a-418c-a64d-bbf49e742189
savefig(plt_nsp, "../../figures/document_model_nsp.pdf")

# ╔═╡ 429651bf-0cc2-4215-abfd-c77d596f3ca6
savefig(plt_base, "../../figures/document_model_base.pdf")

# ╔═╡ e867cc3f-c496-438b-976f-65a8e42409e6
md"""
## Figure: cluster co-occupancy
"""

# ╔═╡ 2c156069-8db4-46c4-88e5-d2882a1d1d13


# ╔═╡ ef1ba9b5-af6d-4006-bf0d-38de232acdfd
md"""
## Figure: spike raster
"""

# ╔═╡ 3b8c5ea9-4050-4544-83ff-460ba1630e90


# ╔═╡ Cell order:
# ╠═c89df6e8-4338-11ec-2338-dd887304f326
# ╟─4d98b0bd-3e93-4fff-9873-8c6474f38ed6
# ╠═3de8b9b4-d60a-47a7-a084-97fde9de4feb
# ╠═086186c4-55e5-46cc-9464-b5c81cde470e
# ╠═8ae7c137-d694-417d-9f93-33484766e1fb
# ╠═41cea1a0-1ac5-4f99-a2df-bc82165f5ebc
# ╟─c21040c8-269a-4239-a4f5-fe2c87074031
# ╠═56b34d1d-81f9-4434-98c2-6185db8fe19a
# ╠═94fe7f2e-d82d-4929-ac3f-2b3e7784ce4f
# ╟─03c03483-c87d-483e-82fe-721099bd89d3
# ╠═c07b33b8-4c65-4da9-99a1-83fd8d499624
# ╠═0fc0e5ee-e631-4c7a-a762-a3d53c4a4268
# ╠═bb7af94b-5779-4f52-bef5-a90169c5fd14
# ╠═b69306f8-6e64-43b0-a4ce-7a4f3c9c2536
# ╟─8f24232e-a7ea-4d87-8a64-9ef23fcac0d6
# ╠═3650e87b-2672-4398-9f04-8240194166d1
# ╟─65d6ece6-61ca-4c9c-a882-fbb3064e54f7
# ╟─6c37769b-f756-4e00-8762-dd669bcb1478
# ╠═f2afe3b5-ed3a-468d-a032-ede27ece64bd
# ╠═f8bf84c1-6372-4456-a6f7-ca6af03c91ed
# ╠═eedf8b14-4e93-4832-85e6-5b2c04cfc385
# ╠═6173a155-f8ff-421d-be52-66c14b349338
# ╠═8c045873-bba5-4842-bbe6-2b6b466a1b53
# ╠═3768c42c-8deb-4fc1-a76b-44e63f533d17
# ╠═d9d7da1d-8c29-4473-94e2-a83b60b10db1
# ╟─01157707-9ff9-4610-8131-0471d9a44a0a
# ╠═0379e63e-7e7a-40be-b319-3224fd1ec12d
# ╠═78e483ff-058b-48c0-9da7-2def7e74dbde
# ╠═6fca003b-d0f9-4d13-873c-68f4c74add38
# ╠═f1a2057e-2cb6-47d8-bd6f-a626af7d0321
# ╠═69edfdf0-077a-442b-91d5-40d51774d943
# ╠═19fa4c42-d92a-4e65-86cf-f68642c1eb5a
# ╠═e54f802e-fda9-4843-8321-a4a1fdc52d3c
# ╟─3e054434-e7ce-4e95-aea7-0229ba192dd7
# ╠═585f7e78-e31e-4fed-8dc4-768070cf959a
# ╠═339ff2db-2757-4d20-8e5d-b6dded9954cc
# ╠═873815b6-9af4-4481-a634-a63d3d755429
# ╠═b6772648-34fa-42f8-8fe5-ec62265950b9
# ╠═bfcee336-a4d7-43a2-904e-b24cc7f161dc
# ╠═13d1fc9e-fff3-40ff-a6ba-b5d60d48016e
# ╠═bb743a40-e9a0-4c22-a54e-44dc4631afb0
# ╠═e6721d4f-d746-4885-9a25-28995d938efa
# ╠═af716196-b8ec-4f8a-bd81-fb78b61be721
# ╠═e6445104-e533-4967-bca8-c5674aa3f490
# ╠═504f4121-c295-4cb4-9b90-5545533a09d9
# ╠═d876cfc9-aba8-4372-a9a2-1c435e1a7257
# ╠═de03c4a3-ec58-46b3-9c34-12f729d0f89c
# ╠═31e14f3c-b2c6-4d5a-b340-087d7c139a06
# ╟─7a645f38-8046-4a2a-b075-dedd6bcfdaf3
# ╠═4637b901-c112-4d13-91c9-0810906cfdd5
# ╠═4a41a034-1822-41fb-8f5a-b450aa649479
# ╠═e25fe7e2-a574-4623-b36b-6c9b79ca6d9a
# ╠═db5723a1-3077-4073-8913-ed62aeca520b
# ╠═e52847bd-81c1-4b61-971c-223929ed3bea
# ╠═f8bf18fb-f444-4234-a0bc-471318391879
# ╠═ce1f0a38-e8cb-45b4-bc15-c9b2c5a06afd
# ╠═f7f93a79-1b7a-418c-a64d-bbf49e742189
# ╠═429651bf-0cc2-4215-abfd-c77d596f3ca6
# ╟─e867cc3f-c496-438b-976f-65a8e42409e6
# ╠═2c156069-8db4-46c4-88e5-d2882a1d1d13
# ╟─ef1ba9b5-af6d-4006-bf0d-38de232acdfd
# ╠═3b8c5ea9-4050-4544-83ff-460ba1630e90
