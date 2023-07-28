using JLD
using PointProcessSequences: get_day_of_week
using Distributions
using StatsPlots
include("../utils.jl")

datadir = "/Users/degleris/data/cables/"

# Load and parse results
if !isdefined(Main, :results)
    results = load(datadir * "month_results.jld", "results")
    assignments = results[:assignments]
    config = results[:config]
    events = results[:events]
    model = results[:model]
    log_p_hist = results[:log_p_hist]
    assignment_hist = results[:assignment_hist]

    min_date = config[:min_date]
    max_date = config[:max_date]
    vocab_cutoff=config[:vocab_offset]

    # Load metadata
    vocab, embassies, dates = load_cables_metadata(datadir)
    word_distr = load_total_word_distribution(datadir)
    data, num_nodes, num_words = load_cables_dataset(
        datadir, min_date, max_date;
        vocab_cutoff=vocab_cutoff, total_word_distr=word_distr
    )
end

# Inspect results
_look(c) = inspect(
    c, min_date, vocab, embassies, dates;
    word_distr=word_distr, num_words=50
)
# _look.(events)

###
# Visualize results
###

# Load relevant variables
start = DATASET_START_DATE + Day(min_date)
stop = DATASET_START_DATE + Day(max_date)
window_length = max_date - min_date

# First, plot the number of cables over time
num_documents(day, k) = sum([(x.position == (day - start).value) && (assignments[i] == k) for (i, x) in enumerate(data)])
xrange = (start - Day(1)):Day(1):(stop + Day(1))
krange = [1, 2, 3, -1]
yrange = zeros(length(xrange), length(krange))
for (col, k) in enumerate(krange)
    yrange[:, col] = num_documents.(xrange, k)
end
data_days = [start + Day(x.position) for x in data]
#yrange = [data_days[assignments .== k] for k in [-1, 1, 2, 3]]
xticks = [xrange[2], xrange[end-1]]

p = groupedbar(
    xrange,
    yrange,
    labels=[1 2 3 "bkgd"],
    xlabel="date",
    ylabel="number of documents",
    bar_position=:stack,
#    bins=length(xrange),
    xticks=(xticks, xticks),
)
# for k in 1:3
#     histogram!(
#         p, data_days[assignments .== k];
#         bins=length(x_range)
#     )
# end
# histogram!(
#     p, y_range[2]
# )

# Plot latent events
events_x = [start + Day(round(Int, e.position)) for e in events]
events_y = [sum(assignments .== e.index) for e in events]

# plot!(p,
#     events_x,
#     events_y,
#     line=:stem,
#     marker=:circle,
#     lw=3,
#     label="latent events",
#     color="red",
#     msw=0.0,
#     ms=7
# )

# Plot Poisson process intensity
function pp_intensity(day, e::CablesEventSummary)
    t = (day - start).value
    D = Normal(e.position, sqrt(e.variance))
    return e.amplitude * pdf(D, t)
end

function pp_intensity(day)
    t = (day - start).value
    d = get_day_of_week(max(0, t))
    if d == 0
        @show t d
    end

    λ0 = model.globals.bkgd_rate * model.globals.day_of_week[d] * 7
    return λ0 + sum([pp_intensity(day, e) for e in events])
end

pp_x = xrange
# pp_y = pp_intensity.(pp_x)
# 
# plot!(p,
#     pp_x,
#     pp_y,
#     label="model intensity",
#     lw=3
# )
# for e in events
#     plot!(p, pp_x, [pp_intensity(x, e) for x in pp_x], label="Event $(e.index)", lw=3)
# end

display(p)

# Plot selected word intensities

