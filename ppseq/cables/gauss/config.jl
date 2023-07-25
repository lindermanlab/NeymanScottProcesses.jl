"""
Parameters for the model used to generate the data.

Requirements
------------
using PointProcessSequences
using LinearAlgebra
"""

_make_seed(word) = prod([Int(c) for c in word])

datadir = "/Users/degleris/data/cables/"
config = Dict(
    :bounds => (1.0, 1.0),
    :max_radius => Inf,
    :event_rate => 4.0,
    :event_amplitude => specify_gamma(30.0, 1e-1),
    :bkgd_amplitude => specify_gamma(30.0, 1e-10),
    :covariance_scale => 8.0 * 1e-3 * [1.0 0.0; 0.0 1.0],
    :covariance_df => 8.0,

    :dataseed => _make_seed("data"),
    :datafile => "gaussian_data.jld",

    :run_seed => 1,
    :run_resultsfile => "gaussian_basic.jld",

    :mask_seed => _make_seed("mask"),
    :mask_resultsfile => "gaussian_mask.jld",

    :comp_seed => 1,
    :comp_resultsfile => "gaussian_comparison.jld",
)


