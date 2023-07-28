"""
Run the cables model on experimental embassy data for the year of 1976.

USAGE
-----
julia cables/year/run.jl
"""

# Imports
using Dates
using PointProcessSequences
include("../utils.jl")

BICENTENNIAL_DATE = Date(1976, 7, 4)
START = Date(1976, 1, 1)
STOP = Date(1976, 12, 31)

# Script parameters
datadir = "/Users/degleris/data/cables/"

seed = parse(Int, ARGS[1])

config = Dict(
    :seed => seed,
    :outfile => "year_results_$(seed).jld",
    
    :min_date => get_dateid(START),
    :max_date => get_dateid(STOP),
    :vocab_offset => 100,
    :max_radius => Inf,

    :event_rate => 1 / 30,
    :event_amplitude => specify_gamma(500, 10^2),
    :bkgd_amplitude => specify_gamma(100, 10^2),
    :variance_scale => 4.0 * 5000.0,
    :variance_pseudo_obs => 5000.0,
    :node_concentration => 1.0,
    :mark_concentration => 1.0,
    :bkgd_node_concentration => 1.0,
    :bkgd_mark_spread => 1.0 ,
    :bkgd_mark_strength => 1e10,
    :bkgd_day_of_week => 1.0 * ones(7),
   
    :max_temp => 100.0,
    :num_anneals => 2,
    :samples_per_anneal => 20,
    :save_every => 1,
    :num_partitions => 4,
)

results = load_train_save(datadir, config)

