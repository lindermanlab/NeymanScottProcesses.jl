using PointProcessSequences
using JLD
using Plots

include("./config.jl");	

dataset = load(datadir * config[:datafile], "dataset")
results = load(datadir * config[:run_resultsfile], "results")

data = dataset[:data]
true_ω = dataset[:assignments]
est_ω = results[:assignments]

color(ω) = (ω == -1) ? "black" : ω
c1(x) = x.position[1]
c2(x) = x.position[2]

make_plot(ω) = scatter(c1.(data), c2.(data), color=color.(ω), legend=false)

p1 = make_plot(true_ω)
p2 = make_plot(est_ω)

plot(p1, p2, layout=(1, 2))
