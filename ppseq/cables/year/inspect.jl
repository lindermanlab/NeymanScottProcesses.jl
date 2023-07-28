using JLD
using PointProcessSequences: get_day_of_week
using Distributions
using StatsPlots

include("../utils.jl")

datadir = "/Users/degleris/data/cables/"

# Load and parse results
if !isdefined(Main, :results)
    results = load(datadir * "year_results_0.jld", "results")
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
interval = Day(7)

# First, plot the number of cables over time
function num_documents(t1, t2, k) 
    
    s1 = (t1 - start).value
    s2 = (t2 - start).value

    all_docs = 1:length(data)
    in_interval = filter(i -> (s1 <= data[i].position <= s2), all_docs)
    in_interval_and_cluster = filter(i -> (assignments[i] == k), in_interval)

    return sum(in_interval_and_cluster)
end

xrange = (start - Day(1)):interval:(stop + Day(1))
krange = append!([-1], collect(1:length(events)))
yrange = zeros(length(xrange), length(krange))
xticks = [xrange[2], xrange[end-1]]

for (col, k) in enumerate(krange)
    for (row, d) in enumerate(xrange)
        yrange[row, col] = num_documents(d, d+interval-Day(1), k)
    end
end

p = groupedbar(
    xrange,
    yrange,
    xlabel="date",
    ylabel="number of documents",
    bar_position=:stack,
    xticks=(xticks, xticks),
    legend=false,
)

# Plot latent events
events_x = [start + Day(round(Int, e.position)) for e in events]
events_y = [sum(assignments .== e.index) for e in events]

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
pp_y = pp_intensity.(pp_x)

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

