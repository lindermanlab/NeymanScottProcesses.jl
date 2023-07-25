import Random
import JLD
import PyPlot: plt

using Profile
using Revise
using SparseArrays
using PointProcessSequences

include("utils.jl")

DATADIR = "/Users/degleris/data/cables/"
TRAIN = "train.jld"

# Load data
min_date = 1.0
max_date = 80.0
data, num_nodes, num_marks = load_cables_dataset(DATADIR, min_date, max_date)

# Script parameters
seed = 0
max_time = max_date
max_radius = Inf
event_rate = 0.01 # one event every 100 days
event_amplitude_mean = 30.0
event_amplitude_var = 10.0^2
bkgd_amplitude_mean = 1e-10
bkgd_amplitude_var = 1e-10
variance_scale = 1.0
variance_pseudo_obs = 1.0
node_concentration = 1e-1 * ones(num_nodes)
mark_concentration = 1e-1
bkgd_node_concentration = 1.0 * ones(num_nodes)
bkgd_mark_concentration = 1.0 * ones(num_marks, num_nodes) 

event_amplitude = specify_gamma(event_amplitude_mean, event_amplitude_var)
bkgd_amplitude = specify_gamma(bkgd_amplitude_mean, bkgd_amplitude_var)  

# Don't use all the documents
max_docs = 500
data = data[1:max_docs]

# Update prior to incorporate model info
bkgd_node_concentration .+= empirical_node_distribution(data, num_nodes)
bkgd_mark_concentration .+= empirical_word_distribution(data, num_nodes, num_marks)

@show length(data)
@show num_marks
@show num_nodes
println()

# Fit model
Random.seed!(seed)
model = CablesModel(
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
    2,  # anneal rounds
    5,  # samples per anneal
    1000.0,  # max temp
    0,
    5.0,
    1;
    verbose=true,
    anneal_background=true
)

# Inspect results
vocab, embassies, dates = load_cables_metadata(DATADIR)


