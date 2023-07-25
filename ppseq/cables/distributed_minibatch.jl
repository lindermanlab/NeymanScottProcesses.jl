import Random
import JLD
import PyPlot: plt

using Profile

using SparseArrays
using PointProcessSequences

DATADIR = "/Users/degleris/data/cables/"
TRAIN = "train.jld"

# Script parameters
mode = :time
seed = 0
min_date = 1.0
max_date = 80.0

max_time = max_date
max_radius = 20.0
event_rate = 0.1 # one event every 10 days
event_amplitude_mean = 30.0
event_amplitude_var = 10.0^2
bkgd_amplitude_mean = 1e-10
bkgd_amplitude_var = 1e-10


variance_scale = 1.0
variance_pseudo_obs = 1.0

event_amplitude = specify_gamma(event_amplitude_mean, event_amplitude_var)
bkgd_amplitude = specify_gamma(bkgd_amplitude_mean, bkgd_amplitude_var)  

# Set seed
Random.seed!(seed)


####
## Load the data
d = JLD.load(DATADIR * TRAIN)

docs = d["docs"]
dates = float.(d["dates"])
nodes = d["nodes"]
doc_word_mat = d["doc_word_mat"]

d = nothing
GC.gc()


####
## Filter nodes by date
relevant_docs = (min_date .<= dates .<= max_date)

docs = docs[relevant_docs]
dates = dates[relevant_docs]
nodes = nodes[relevant_docs]
doc_word_mat = doc_word_mat[relevant_docs, :]

GC.gc()


####
## Finally, convert all these to cables
data = Cable[]
for i = 1:length(docs)
    push!(data, Cable(dates[i], nodes[i]+1, doc_word_mat[i, :]))
end

max_docs = 5000
data = data[1:max_docs]


####
# Add additional parameters

num_nodes = maximum(nodes) + 1
num_marks = size(doc_word_mat, 2)

node_concentration = 1e-1 * ones(num_nodes)
mark_concentration = 1e-1 * ones(num_marks)

bkgd_node_concentration = 1.0 * ones(num_nodes)
bkgd_mark_concentration = 1.0 * ones(num_marks, num_nodes)  


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
num_partitions = 4
model = make_distributed(primary_model, num_partitions)

fit(data) = annealed_gibbs!(
    model,
    data,
    fill(-1, length(data)),
    2,
    5,
    100.0,
    0,
    2.0,
    1,
    verbose=true
)

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
fit(data[1:10])
println()

# Now fit all the data
Profile.clear()
@time fit(data)


