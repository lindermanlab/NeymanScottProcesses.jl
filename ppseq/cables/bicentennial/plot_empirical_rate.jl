using StatsPlots
include("../utils.jl")

datadir = "/Users/degleris/data/cables/"
min_date = 0.0
max_date = Inf
data, num_nodes, num_words = load_cables_dataset(datadir, min_date, max_date)

data_dates = [x.position for x in data]

xrange = DATASET_START_DATE : Day(1) : DATASET_STOP_DATE
xticks = 1:365:length(xrange)
xticklabels = xrange[1:365:end]

histogram(
    data_dates,
    bins=100,
    legend=false,
    title="empirical rate",
    xlabel="day",
    xticks=(xticks, xticklabels),
)
