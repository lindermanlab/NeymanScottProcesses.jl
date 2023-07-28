"""
Run the cables model on full experimental embassy dataset.

Tips
----
* use `julia -t 4` so start Julia with 4 threads. On older versions of Julia, you might
have to use -p instead.

"""

println("Script initiated.")

# Imports
using Dates
using PointProcessSequences
include("../utils.jl")

println("Packages loaded.")

datadir = ARGS[1]

# Script parameters
config = Dict(
    :seed => 0,
    :outfile => "cables_full.jld",
    :min_date => 450.0, 
    :max_date => get_dateid(DATASET_STOP_DATE) - 300.0,
    :vocab_offset => 100,
    :max_radius => 30.0,

    :event_rate => 1 / 30,
    :event_amplitude => specify_gamma(500, 10^2),
    :bkgd_amplitude => specify_gamma(100, 10^2),
    :variance_scale => 50.0,
    :variance_pseudo_obs => 10.0,
    :node_concentration => 1.0,
    :mark_concentration => 1.0,
    :bkgd_node_concentration => 1.0,
    :bkgd_mark_spread => 1.0 ,
    :bkgd_mark_strength => 1e10,
    :bkgd_day_of_week => 1.0 * ones(7),
   
    :max_temp => 100.0,
    :num_anneals => 2,
    :samples_per_anneal => 50,
    :save_every => 1,
    :num_partitions => 4,
)

results = load_train_save(datadir, config)
