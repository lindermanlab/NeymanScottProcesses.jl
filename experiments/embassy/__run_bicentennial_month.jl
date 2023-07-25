"""
Runs the cables model on experimental embassy data for June and July of 1976.
"""

# Runs
# bicentennial_month_run1_dp_strong.jld

mode = :inspect

@assert mode in (:inspect, :run)




using Dates
using NeymanScottProcesses
using NeymanScottProcesses: get_day_of_week
using Distributions
include("dataset_utils.jl")




# Find data file path
datadir = "/Users/degleris/data/cables"
if datadir === nothing
    if length(ARGS) > 0
        datadir = ARGS[1]
    else
        print("Enter data directory: ")
        datadir = readline()
    end
end


# Script parameters
config = Dict(
    :min_date => get_dateid(Date(1976, 6, 21)),
    :max_date => get_dateid(Date(1976, 7, 31)),
    :vocab_cutoff => 100,
    :percent_masked => 0.05,

    :max_cluster_radius => Inf,
    :cluster_rate => 1.0 / 30,
    :cluster_amplitude => specify_gamma(500, 10^2),
    :cluster_width => specify_inverse_gamma(2.0, (1e-4)^2),
    :background_amplitude => specify_gamma(1000, 10^2),
    :background_word_concentration => 1e8,
    :background_word_spread => 1.0,

    :seed => 1976,
    :samples_per_mask => 5,
    :masks_per_anneal => 3,
    :save_interval => 1,
    :temps => exp10.(vcat(range(6.0, 0.0, length=5))),
    #exp10.(vcat(range(6.0, 0, length=20), fill(0.0, 3))),
    :results_path => "bicentennial_month_run1_dp_strong.jld",
)



# Run mode - generate results
if mode == :run
    results, model = load_train_save(datadir, config)
end




# Inspect mode - just look at results
data, word_distr, meta, config, results, model = load_results(datadir, config)

# Mask data
Random.seed!(12345)
masks = create_random_mask(model, 1.0, config[:percent_masked])
masked_data, unmasked_data = split_data_by_mask(data, masks)




# Some useful data
normalized_word_distr = word_distr ./ sum(word_distr, dims=1)
cluster_inds = sort(unique(last(results.assignments)))




# Every debugging tool ever
inspect(cluster) = inspect(cluster, get_date(config[:min_date]), meta;
    num_words=50, num_embassies=5, word_distr=normalized_word_distr)

get_doc(k, j) = inspect(data[last(results.assignments) .== k][j], 
    config[:min_date], meta; num_words=50)

get_delta(a1, a2) = count(x->(x != 0), a1 - a2)

get_deltas(a_hist) = [get_delta(a_hist[i], a_hist[i-1]) for i in 2:length(a_hist)]

num_clusters(ω) = length(unique(ω)) - 1

pc_background(a_hist) = [
    sum([a_hist[i][j] == -1 for i in 1:length(a_hist)])
    for j in 1:length(last(a_hist))
]

move_freq(a_hist) = [
    sum([a_hist[i][j] != a_hist[i-1][j] for i in 2:length(a_hist)])
    for j in 1:length(last(a_hist))
]

function get_avg_doc_length(k)
    cluster_data = data[last(results.assignments) .== k]
    return mean([sum(x.words) for x in cluster_data])
end

function model_intensity(model, word, t; meta=meta)
    word_ind = findfirst(w -> w == word, meta.vocab[:, :word])
    G = model.globals

    # Add background intensity
    λ = (
        G.bkgd_rate 
        * G.bkgd_word_probs[word_ind, :]' * G.bkgd_embassy_probs 
        * G.day_of_week_probs[get_day_of_week(t)]
    )

    # Add cluster intensity
    for C in model.clusters
        Dt = Normal(C.sampled_position, C.sampled_variance)
        λc = C.sampled_amplitude * pdf(Dt, t)
        λc *= C.sampled_word_probs[word_ind]

        λ += λc
    end

    return λ
end

clusters = [C for C in model.clusters]


ts = [c.position for c in data]
t_grid = minimum(ts) : 0.1 : maximum(ts)
keywords = ["ENTEBBE", "HOSTAGE", "BICENTENNIAL", "CONGRATULATIONS"]

word_intensity = w -> [model_intensity(model, w, t) for t in t_grid]
total_intensity = C -> [pdf(Normal(C.sampled_position, C.sampled_variance), t) * C.sampled_amplitude for t in t_grid]

λs = word_intensity.(keywords)


Δω_hist = get_deltas(results.assignments)
K_hist = num_clusters.(results.assignments)
bkgd_hist = pc_background(results.assignments)
move_hist = move_freq(results.assignments)



# histogram(bkgd_hist, legend=false, nbins=50, xlabel="percent of samples in background cluster", 
#   normalize=true, ylabel="frequency")

# histogram(move_hist, legend=false, nbins=50, xlabel="number of assignment moves", 
#   normalize=true, ylabel="frequency")

println("\007")


# Make cable intensity plot
theme(:default, label=nothing)

plt = plot()
#ylabel="frequency", yguidefontsize=9, ytickfontsize=9, yguidefont="times", ytickfont="times")
plot!(xtickfontsize=8, xguidefont="times", ytickfontsize=8, ytickfont="times", yticks=nothing)
relevant_colors = [:Purple, :ForestGreen]
for (i, C) in enumerate(clusters)
    relevant_words = inspect(C)
    max_height = C.sampled_amplitude * pdf(Normal(0, C.sampled_variance), 0)
    fontsize = 6

    
    if i ∈ [2, 4]
        shift = (i == 4) ? -6 : 6
        line_color = pop!(relevant_colors)
        relevant_words = filter(w -> length(w) > 3, relevant_words)
        annotation = text(join(relevant_words[1:5], "\n"), fontsize, :Bold, color=line_color)
        annotate!(plt, C.sampled_position + shift, max_height + 30, annotation, color=line_color)
    else
        line_color = :LightGray
    end

    plot!(plt, t_grid, total_intensity(C), c=line_color)
end

plot!(plt, size=(300, 100))
savefig(plt, "embassy_results.pdf")
display(plt)
