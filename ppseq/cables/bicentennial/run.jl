"""
Run the cables model on experimental embassy data.

USAGE
-----
julia cables/bicentennial/run.jl
"""

# Imports
using Dates
using PointProcessSequences
include("../utils.jl")

BICENTENNIAL_DATE = Date(1976, 7, 4)
mode = :anthony

if mode == :anthony
    datadir = "/Users/degleris/data/cables/"
elseif mode == :yixin
    datadir = "/rigel/stats/users/yw2539/data/cables_julia/"
end

# Script parameters
config = Dict(
    :seed => 0,
    :outfile => "month_results.jld",
    
    :min_date => get_dateid(BICENTENNIAL_DATE - Day(13)),
    :max_date => get_dateid(BICENTENNIAL_DATE + Day(21)),
    :vocab_offset => 100,
    :max_radius => Inf,

    :event_rate => 1 / 30,
    :event_amplitude => specify_gamma(500, 10^2),
    :bkgd_amplitude => specify_gamma(100, 10^2),
    :variance_scale => 4.0 * 10_000.0,
    :variance_pseudo_obs => 10_000.0,
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
    :num_partitions => 1,
)

results = load_train_save(datadir, config)

#min_doclen = 20.0
#max_doclen = 200.0
# Filter cables by length
# TODO
# doc_lengths = sum(doc_word_mat, dims=2)
# longdocs = (min_doclen .<= doc_lengths .<= max_doclen )[:,1]
# docs = docs[longdocs]
# dates = dates[longdocs]
# nodes = nodes[longdocs]
# doc_word_mat = doc_word_mat[longdocs, :]
# droptol!(doc_word_mat::SparseMatrixCSC, 1e-6) # drop entries with zero value for SparseArrays

# top words each day
# TODO
# for date in 1:(max_date-min_date)
#     println("date")
#     println(date)
#     datespec_cables = doc_word_mat[dates.==date, :]
#     datespec_wordct = sum(datespec_cables, dims=1)
#     datespec_freqwords = sortperm(datespec_wordct[1,:], rev=true)[1:100]
#     println(vocab["word"][datespec_freqwords])
#     println(datespec_wordct[3865]) # word count: bicentennial
#     println(datespec_wordct[668]) # word count: lebanon
# end
