"""
Runs the cables model on experimental embassy data for June and July of 1976.
"""

mode = :inspect

@assert mode in (:inspect, :run)




using Dates
using NeymanScottProcesses
include("dataset_utils.jl")




# Find data file path
if length(ARGS) > 0
    datadir = ARGS[1]
else
    print("Enter data directory: ")
    datadir = readline()
end
println("Data directory set to $datadir")




# Script parameters
config = Dict(
    :min_date => get_dateid(Date(1976, 6, 21)),
    :max_date => get_dateid(Date(1976, 7, 31)),
    :vocab_cutoff => 100,

    :max_cluster_radius => Inf,
    :cluster_rate => 1.0 / 30,
    :cluster_amplitude => specify_gamma(500, 10^2),
    :cluster_width => specify_inverse_gamma(2.0, 0.01^2),
    :background_amplitude => specify_gamma(100, 10^2),
    :background_word_concentration => 1000.0,
    :background_word_spread => 1.0,

    :seed => 1976,
    :samples_per_anneal => 10,
    :save_interval => 1,
    :temps => exp10.(vcat(range(6.0, 0.0, length=5)))
    #exp10.(vcat(range(6.0, 0, length=20), fill(0.0, 3))),
    :results_path => "bicentennial_month.jld",
)



# Run mode - generate results
if mode == :run
    results, model = load_train_save(datadir, config)
end
# Inspect mode - just look at results
data, word_distr, meta, results, model = load_results(datadir, config)