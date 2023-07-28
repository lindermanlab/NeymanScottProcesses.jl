import JLD
using PyPlot
using StatsBase: mean


# Loads runtimes, data_sizes, cluster_size
results = JLD.load((@__DIR__) * "/results.jld")
runtimes = results["runtimes"]
data_sizes = results["data_sizes"]

# Process data
num_seed, num_time, num_threads = size(data_sizes)
avg_data_size = reshape(mean(data_sizes, dims=1), :, num_threads)
avg_runtime = reshape(mean(runtimes, dims=1), :, num_threads) / 60


# Make a plot of runtimes
plt.figure()
for nthread in 1:size(avg_data_size, 2)
    label = string(nthread) * " threads"
    (nthread == 1) && (label = label[1:end-1])
    plt.plot(
        avg_data_size[:, nthread], 
        avg_runtime[:, nthread],
        label=label,
        "o-", ms=8, lw=3
    )
    
end

plt.xlabel("number of documents")
plt.ylabel("minutes per 100 Gibbs samples")
plt.grid(true)
plt.title("synthetic cable data performance")
plt.legend()
plt.savefig((@__DIR__) *  "/figure.png")
