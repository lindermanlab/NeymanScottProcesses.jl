using JLD: @save

include("../utils.jl")

# Change this on your machine
datadir = "/Users/degleris/data/cables/"

println("Loading full dataset")
data, num_nodes, num_words = load_cables_dataset(datadir, 0, Inf)

println("Computing empirical word distribution for $(length(data)) documents")
word_distr = empirical_word_distribution(data, num_nodes, num_words)

path = datadir * "empirical_word_distr.jld"
@save path word_distr
