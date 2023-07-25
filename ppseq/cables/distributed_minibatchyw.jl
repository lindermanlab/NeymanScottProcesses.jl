import Random
import JLD
# import PyPlot: plt

using Profile

using SparseArrays
using PointProcessSequences

using Statistics

using CSV, DataFrames

include("utils.jl")

# julia -t 4 scriptname.jl

# start julia with julia -p 4

# include("distributed_minibatchyw.jl")

DATADIR = "/rigel/stats/users/yw2539/data/cables_julia/"
TRAIN = "train.jld"

# start_date = date(1973, 1, 1)
# stop_date = date(1978, 12, 31)
# bicentennial_date = date(1976, 7, 4)

# node id not shifted by one
# vocab dates are the original indices added one (julia indices start from one)


# Script parameters
# mode = :time #benchmark/time code
seed = 0
min_date = 1270.0 # 1270.0 # 1976-06-14 maybe we can go even earlier
max_date = 1300.0 # 1290.0 # 1976-07-14

min_doclen = 20.0
max_doclen = 200.0
vocab_offset = 100

# Set seed
Random.seed!(seed)


####
## Load the data
d = JLD.load(DATADIR * TRAIN)

docs = d["docs"]
dates = float.(d["dates"])
nodes = d["nodes"]
doc_word_mat = d["doc_word_mat"]

vocab = DataFrame(CSV.File("/rigel/stats/users/yw2539/data/cables_pro_daily/daily/vocab.dat", header=[:word]))



# truncate vocab
cables_wordct = sum(doc_word_mat, dims=1) 
cables_wordfreq = sortperm(cables_wordct[1,:], rev=true)
# println(vocab["word"][cables_wordfreq][1:200])
invalid_word_idxs = cables_wordfreq[1:vocab_offset]
doc_word_mat[:, invalid_word_idxs] .= 0.


# compute background word count from the complete cables (without subsetting)
alldata = Cable[]
num_nodes = maximum(nodes) + 1
num_marks = size(doc_word_mat, 2)
for i = 1:length(docs)
    push!(alldata, Cable(dates[i], nodes[i]+1, doc_word_mat[i, :]))
end
nodespec_markdist = empirical_word_distribution(alldata, num_nodes, num_marks)
# can we save this nodespec_markdist to a file so that we can just load it everytime without recomputing it?
alldata = nothing

# d = nothing
GC.gc()

####
## Filter nodes by date
relevant_docs = (min_date .<= dates .<= max_date) 

docs = docs[relevant_docs]
dates = dates[relevant_docs] .- min_date
nodes = nodes[relevant_docs]
doc_word_mat = doc_word_mat[docs, :]

## Filter cables by length
doc_lengths = sum(doc_word_mat, dims=2)
longdocs = (min_doclen .<= doc_lengths .<= max_doclen )[:,1]
docs = docs[longdocs]
dates = dates[longdocs]
nodes = nodes[longdocs]
doc_word_mat = doc_word_mat[longdocs, :]
droptol!(doc_word_mat::SparseMatrixCSC, 1e-6) # drop entries with zero value for SparseArrays


# top words each day
for date in 1:(max_date-min_date)
    println("date")
    println(date)
    datespec_cables = doc_word_mat[dates.==date, :]
    datespec_wordct = sum(datespec_cables, dims=1)
    datespec_freqwords = sortperm(datespec_wordct[1,:], rev=true)[1:100]
    println(vocab["word"][datespec_freqwords])
    println(datespec_wordct[3865]) # word count: bicentennial
    println(datespec_wordct[668]) # word count: lebanon
end

# compute mean background marks for each node

# for node in 1:num_allnodes
#     println(node)
#     nodespec_cables = float(doc_word_mat[nodes.==node, :])
#     # normalize every row of the matrix to compute probability
#     for (row,s) in enumerate(sum(nodespec_cables,dims=2))
#         s == 0 && continue
#         nodespec_cables[row, :] = nodespec_cables[row, :]/s
#     end
#     nodespec_prob = mean(nodespec_cables, dims=1)
#     meanbgmarks[node,:] = nodespec_prob
# end


GC.gc()


####
## Finally, convert all these to cables
data = Cable[]
for i = 1:length(docs)
    push!(data, Cable(dates[i], nodes[i]+1, doc_word_mat[i, :]))
end

# max_docs = 5000
# data = data[1:max_docs]


max_time = max_date - min_date # max time period
max_radius = Inf # truncation boundary of time of events, can be Inf
event_rate = 1/30. # one event every x days
event_amplitude_mean = 500
event_amplitude_var = 10^2
bkgd_amplitude_mean = 100 # prior on number of background events
bkgd_amplitude_var = 10^2


variance_scale = 10.0 # width of event time, parameters of inverse gamma, beta parameter
variance_pseudo_obs = 2.0 # 1d NIW prior on variance of event times

event_amplitude = specify_gamma(event_amplitude_mean, event_amplitude_var)
bkgd_amplitude = specify_gamma(bkgd_amplitude_mean, bkgd_amplitude_var)  

####
# Add additional parameters

num_nodes = maximum(nodes) + 1
num_marks = size(doc_word_mat, 2)

node_concentration = 1.0 * ones(num_nodes)
mark_concentration = 1.0

# node_concentration .+= 1e1 * empirical_node_distribution(data, num_nodes)

bkgd_node_concentration = 1.0 * ones(num_nodes)
bkgd_mark_concentration = 1.0 * ones(num_marks, num_nodes)  

# Update prior to incorporate model info
# bkgd_node_concentration .+= 1e5 * empirical_node_distribution(data, num_nodes)
bkgd_mark_concentration .+= 1e10 * nodespec_markdist



@show length(data)
@show num_marks
@show num_nodes
println()

####
# Fit model
primary_model = CablesModel(
    max_time,
    max_radius,
    CablesPriors(
        event_rate,
        event_amplitude,
        bkgd_amplitude,

        node_concentration,
        mark_concentration,
        bkgd_node_concentration,
        bkgd_mark_concentration,
    
        variance_scale,
        variance_pseudo_obs,
    )
)

docs, dates, nodes, doc_word_mat = nothing, nothing, nothing, nothing
GC.gc()

# Make distributed
num_partitions = 8
model = make_distributed(primary_model, num_partitions)

# fit(data) = annealed_gibbs!(
#     model,
#     data,
#     fill(-1, length(data)),
#     2, # number of anneals
#     5, # samples per anneal
#     100.0, # max temp
#     0, # number of split merge moves
#     2.0, # split merge radius
#     1, # how often to save data in terms of samples
#     verbose=true
# )


(
    assignments,
    assignment_hist,
    log_p_hist,
    latent_event_hist,
    globals_hist
) = annealed_gibbs!(
    model,
    data,
    fill(-1, length(data)),
    2, # number of anneals
    5, # samples per anneal
    100.0, # max temp
    0, # number of split merge moves
    2.0, # split merge radius
    1, # how often to save data in terms of samples
    verbose=true
);


# Q: as i increase the number of anneals and samples, the jobs get
# killed.


# latent_event_hist

# fieldnames(typeof(globals_hist[end]))
# (:bkgd_rate, :bkgd_node_prob, :bkgd_mark_prob)

# sortperm(latent_event_hist[end][1].markproba, rev=true)

# gibbs_sample!(
#     model,
#     data,
#     fill(-1, length(data)),
#     10,  # num samples
#     0,
#     5.0,
#     1;
#     verbose=true,
# )


# First fit a tiny dataset
# fit(data)
# println()


# # Imports
# using CSV, DataFrames
# # Load file
# vocab = DataFrame(CSV.File("/rigel/stats/users/yw2539/data/cables_pro_daily/daily/unique_words.txt", header=[:word]))
# # Index
# data[1, "word"]  # returns "VISIT"
# data["word"][1] # returns "VISIT"

# Now fit all the data
# Profile.clear()
# @time fit(data)


# need to sort based on how many cables assigned to each event

println([sum(assignments .== k) for k in sort(unique(assignments))])
clusters_size = [sum(assignments .== k) for k in sort(unique(assignments))][2:end]
clusters = sortperm(clusters_size, rev=true)


# Inspect results
RAW_DATADIR = "/rigel/stats/users/yw2539/data/cables_pro_daily/daily/"
vocab, embassies, dates = load_cables_metadata(RAW_DATADIR)

for i in clusters
    event = latent_event_hist[end][i]
    println("\n\n\n event")
    # println(event)
    println("time")
    println(event.position)
    freqwords = sortperm(event.markproba, rev=true)[1:25]
    println(vocab["word"][freqwords])
    inspect(latent_event_hist[end][i], min_date, vocab, embassies, dates, num_words=20, num_embassies=5)
    println("# cables assigned")
    println(sum(assignments .== i))
end

println([sum(assignments .== k) for k in sort(unique(assignments))])
