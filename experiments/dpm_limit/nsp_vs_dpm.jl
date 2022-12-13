### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 6345af40-f59f-4bbb-a2a5-0fa08dbf5393
using DrWatson

# ╔═╡ 73aead64-1789-41b7-9fb8-4095a8d000f5
@quickactivate

# ╔═╡ acc93e48-e012-11eb-048e-35c010d6acee
using Revise, NeymanScottProcesses

# ╔═╡ 910bd6b8-db64-42bc-a15c-ec428dd4611c
using Random

# ╔═╡ 459c22a2-1f74-4226-895b-26aa0140298e
using LinearAlgebra: I

# ╔═╡ baa97d5f-0098-4bef-8957-c152dda2ee25
using Plots, StatsPlots

# ╔═╡ 82c80d62-2cbc-4c3f-89d9-eba8d2bc69e4
using PlutoUI

# ╔═╡ 6aabf992-6a77-4f0d-bdaf-962ec9ad69ca
TableOfContents()

# ╔═╡ 5de272b0-931a-4851-86ac-4249860a9922
theme(
    :default, label=nothing, 
    tickfont=font(:Times, 8), guidefont=(:Times, 8), titlefont=(:Times, 8), 
    legendfont=(:Times, 8),
    ms=4, msw=0.25, colorbar=false, grid=false, frame=:box,
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
    num_chains = 1

	data_seed = 1
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

# ╔═╡ fd006bd8-1110-43a0-a4b2-d23eb529eb37
function generate_data()
	# Set seed
    Random.seed!(data_seed)

	# Build priors
	priors = GaussianPriors(η, Ak, A0, Ψ, ν)

	# Build model
    gen_model = GaussianNeymanScottModel(bounds, priors)

	# Sample data
    data, assignments, clusters = sample(gen_model; resample_latents=true)
    data = Vector{RealObservation{2}}(data)

    return @strdict(priors, gen_model, data, assignments, clusters)
end

# ╔═╡ 751f3312-f20c-45e4-abfe-e4ccf5d238a4
observation_data = generate_data();

# ╔═╡ 55f5511f-b199-4deb-b82e-2a6091c8b06e
@unpack priors, gen_model, data, clusters, assignments = observation_data

# ╔═╡ 10924047-b95e-41db-86ef-6f5c2cbea1a5
length(data)

# ╔═╡ 5d42306d-55d3-418f-9603-a217730bf653
data_x = [x.position[1] for x in data]; data_y = [x.position[2] for x in data]

# ╔═╡ 3e7a2f7e-7068-47b5-a855-f5a77e83cc19
md"""
### Plot True Data
"""

# ╔═╡ 91ce2bdc-3095-44a8-a7e0-3a2d5e0625a8
function plot_clusters!(plt, clusters)
	for C in clusters
		covellipse!(
			plt, C.sampled_position, C.sampled_covariance, 
			n_std=3, aspect_ratio=1, 
			alpha=0.3, c=1
		)
	end
end

# ╔═╡ 28075f70-b967-4fa6-95a1-cddb20782076
function make_data_plot(data_x, data_y)
	plt = plot(xticks=nothing, yticks=nothing, xlim=(0, 1), ylim=(0, 1),
	frame=:box)
	scatter!(data_x, data_y, c=:black, ms=1.0, alpha=0.5)
	return plt
end

# ╔═╡ 31444176-7908-4eef-865d-4096aed328cd
plt_true_data = let
    # Format plot for data
    plt = make_data_plot(data_x, data_y)
	plot_clusters!(plt, clusters)

	plot!(size=(200, 200), title="True (NSP)")
	
	plt
end

# ╔═╡ d98caa8b-0c20-4b41-b3e2-404061a6f575
md"""
## Fit data with NSP and DPM
"""

# ╔═╡ 60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# Construct samplers
begin
    sampler = GibbsSampler(num_samples=1000, save_interval=1, num_split_merge=10, verbose=false, split_merge_gibbs_moves=1)
end;

# ╔═╡ 5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
md"""
### NSP
"""

# ╔═╡ 0a91fe9a-1f4e-4b10-beef-896d41fcadd3
begin
    nsp_model = []
    r_nsp = []
    
    t_nsp = @elapsed for chain in 1:num_chains
        Random.seed!(chain)
        model = GaussianNeymanScottModel(bounds, priors)

		z0 = rand(1:length(data), length(data))
        r = sampler(model, data, initial_assignments=z0)
        
        push!(nsp_model, model)
        push!(r_nsp, r)
    end
    
    "Fit $num_chains models in $t_nsp seconds"
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
    plt_fit_data_nsp = make_data_plot(data_x, data_y)
	plot_clusters!(plt_fit_data_nsp, nsp_model[1].clusters)
	plot!(title="Learned (NSP)")
    
	#nsp_ω = make_consistent(r_nsp[2].assignments[end], data_x)
    #scatter!(data_x, data_y, c=nsp_ω, title="fit with nsp")
    
    plot(plt_true_data, plt_fit_data_nsp, size=(500, 200))
end

# ╔═╡ 65fbe8af-0c79-4710-a7a2-9b5b1b456a42
md"""
### DPM
"""

# ╔═╡ 52a0caa9-ca84-401b-bddf-c3398ffa9bf4
begin
    Random.seed!(1)
    
    scaling = 1e6
    
    # Reset priors 
    dpm_Ak = RateGamma(Ak.α / scaling, Ak.β)
    dpm_η = η * scaling
    dpm_priors = GaussianPriors(dpm_η, dpm_Ak, A0, Ψ, ν)
    
    dpm_model = []
    r_dpm = []
    
    t_dpm = @elapsed for chain in 1:num_chains
		Random.seed!(chain)
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
    plt_fit_data_dpm = make_data_plot(data_x, data_y)
	plot_clusters!(plt_fit_data_dpm, dpm_model[1].clusters)
	plot!(title="Learned (DPM)")
	
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

# ╔═╡ 84170512-8fd9-430f-9124-f53bbe2a0e8e
md"""
### Violin Plot: Number of Clusters
"""

# ╔═╡ e571a594-4a6e-4dd5-8762-f7330b3707ce
begin
    Random.seed!(11)
    
    num_cluster_true = length(clusters)
    
    # Set up plot
    plt_num_clusters = plot(title="Number of Clusters")
    plot!(xticks=(1:2, ["NSP", "DPM"]))
    plot!(size=(200, 200))
    #plot!(ylim=(0, Inf))
	plot!(yticks=0:20:100)
    plot!(legend=:topleft, grid=false)

    # Plot true number of clusters
    hline!(0.5:0.01:2.5, [num_cluster_true], lw=2, color=3, alpha=0.5, label="True")
        
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
        
        dotplot!(num_clus_x, num_cluster, marker=(:Black, 1), alpha=0.5)
    end 
    
    # Save
    save_and_show(plt_num_clusters, "num_clusters")
end

# ╔═╡ 21439aeb-b306-4e44-b566-934aa6c3adb5
md"""
### Heatmap: Co-occupancy
"""

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

# ╔═╡ c8a901f5-84b7-4eae-9f4c-4d5e74e9c02b
md"""
### Violin Plot: Co-occupancy
"""

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
    plt_acc = plot(title="Co-occupancy Accuracy", xticks=(1:2, ["NSP", "DPM"]))
    plot!(size=(200, 200))
	plot!(grid=false)
    
    # Plot data
    violin!(acc_x, acc_y)
    boxplot!(acc_x, acc_y, fillalpha=0.5, outliers=false)
    dotplot!(acc_x, acc_y, marker=(:Black, 1), alpha=0.5)

	plot!(yticks=0.95:0.01:1.0, ylim=(0.96, 1.0))
    
    save_and_show(plt_acc, "accuracy")
end

# ╔═╡ efc4144b-3578-4c91-9575-04a8f98b6816
md"""
## Final Plot
"""

# ╔═╡ 4fb64ab7-b329-451c-b4ff-9ae80ff6ae59
begin	
    plt_everything = plot(
        plt_true_data, plt_fit_data_nsp, plt_fit_data_dpm, 
        plt_acc, plt_num_clusters, 
        layout=(1, 5), size=(650, 130), dpi=200
    )
    
    save_and_show(plt_everything, "full")
end

# ╔═╡ Cell order:
# ╠═6345af40-f59f-4bbb-a2a5-0fa08dbf5393
# ╠═73aead64-1789-41b7-9fb8-4095a8d000f5
# ╠═acc93e48-e012-11eb-048e-35c010d6acee
# ╠═910bd6b8-db64-42bc-a15c-ec428dd4611c
# ╠═459c22a2-1f74-4226-895b-26aa0140298e
# ╠═baa97d5f-0098-4bef-8957-c152dda2ee25
# ╠═82c80d62-2cbc-4c3f-89d9-eba8d2bc69e4
# ╠═6aabf992-6a77-4f0d-bdaf-962ec9ad69ca
# ╠═5de272b0-931a-4851-86ac-4249860a9922
# ╠═be352c60-88b9-421a-8fd7-7d34e19665e6
# ╟─d1d53b74-3e7b-44eb-b0e7-1e4b612edeb2
# ╠═b0114a0b-58e4-44d2-82db-3a6131435b32
# ╟─b9ee61fa-c387-404b-b273-11dcfa8b63a0
# ╟─fd006bd8-1110-43a0-a4b2-d23eb529eb37
# ╠═751f3312-f20c-45e4-abfe-e4ccf5d238a4
# ╠═55f5511f-b199-4deb-b82e-2a6091c8b06e
# ╠═10924047-b95e-41db-86ef-6f5c2cbea1a5
# ╠═5d42306d-55d3-418f-9603-a217730bf653
# ╟─3e7a2f7e-7068-47b5-a855-f5a77e83cc19
# ╟─91ce2bdc-3095-44a8-a7e0-3a2d5e0625a8
# ╠═28075f70-b967-4fa6-95a1-cddb20782076
# ╠═31444176-7908-4eef-865d-4096aed328cd
# ╟─d98caa8b-0c20-4b41-b3e2-404061a6f575
# ╠═60cf6826-5b63-4d0d-8ee9-1c78b6e5b5dc
# ╟─5af966fc-cf8e-4a54-9eb3-c84c445ad6f0
# ╠═0a91fe9a-1f4e-4b10-beef-896d41fcadd3
# ╠═b4180fdc-b209-414e-a028-b7890e69c302
# ╠═b301bb90-2178-4d49-bca2-e1f7ce59975f
# ╟─27a3553e-9211-45b8-b963-55c4511e6917
# ╟─65fbe8af-0c79-4710-a7a2-9b5b1b456a42
# ╠═52a0caa9-ca84-401b-bddf-c3398ffa9bf4
# ╟─53229684-46ec-49ac-9c03-78bcaa636165
# ╠═17406fcc-ff2e-4fef-a552-06063ec70872
# ╟─77f2fb41-c0bc-4aac-a98d-0048c7b2a8b0
# ╠═e6901d7d-6f3f-493e-9d00-7524475c5ccb
# ╟─84170512-8fd9-430f-9124-f53bbe2a0e8e
# ╠═e571a594-4a6e-4dd5-8762-f7330b3707ce
# ╟─21439aeb-b306-4e44-b566-934aa6c3adb5
# ╟─f3ae22fc-0bb6-4469-ac9d-2bc32252c1a3
# ╟─c8a901f5-84b7-4eae-9f4c-4d5e74e9c02b
# ╠═cd0c0109-0dee-4da8-b9c0-282ec378bf63
# ╟─efc4144b-3578-4c91-9575-04a8f98b6816
# ╠═4fb64ab7-b329-451c-b4ff-9ae80ff6ae59
