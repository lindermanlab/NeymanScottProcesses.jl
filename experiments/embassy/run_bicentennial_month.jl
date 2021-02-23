"""
Runs the cables model on experimental embassy data for June and July of 1976.
"""

mode = :run

@assert mode in (:inspect, :run)




using Dates
using NeymanScottProcesses
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

    :max_cluster_radius => Inf,
    :cluster_rate => 1.0 / 30,
    :cluster_amplitude => specify_gamma(500, 10^2),
    :cluster_width => specify_inverse_gamma(2.0, (1e-4)^2),
    :background_amplitude => specify_gamma(1000, 10^2),
    :background_word_concentration => 10*20_000,  # 10_000 --- 1e8
    :background_word_spread => 1.0,

    :seed => 1976,
    :samples_per_anneal => 10,
    :save_interval => 1,
    :temps => exp10.(vcat(range(6.0, 0.0, length=5))),
    #exp10.(vcat(range(6.0, 0, length=20), fill(0.0, 3))),
    :results_path => "bicentennial_month.jld",
)



# Run mode - generate results
if mode == :run
    results, model = load_train_save(datadir, config)
end




# Inspect mode - just look at results
data, word_distr, meta, results, model = load_results(datadir, config)




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


Δω_hist = get_deltas(results.assignments)
K_hist = num_clusters.(results.assignments)
bkgd_hist = pc_background(results.assignments)
move_hist = move_freq(results.assignments)

# histogram(bkgd_hist, legend=false, nbins=50, xlabel="percent of samples in background cluster", 
#   normalize=true, ylabel="frequency")

# histogram(move_hist, legend=false, nbins=50, xlabel="number of assignment moves", 
#   normalize=true, ylabel="frequency")

println("\007")