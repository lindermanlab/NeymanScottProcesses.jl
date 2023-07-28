using PointProcessSequences
using JLD: @save

include("gibbs.jl")

# Set parameters for roughly 2 cables per second
event_amplitude = 30.0
event_rate = 1.0 * (1/event_amplitude)
bkgd_amplitude = 1.0

seeds = collect(1 : 5)
times = collect(500.0 : 500.0 : 2500.0)
threads = collect(1 : 4)

num_seeds = length(seeds)
num_times = length(times)
num_threads = length(threads)

runtimes = zeros(num_seeds, num_times, num_threads)
data_sizes = zeros(num_seeds, num_times, num_threads)
cluster_sizes = zeros(num_seeds, num_times, num_threads)

for (ind3, nthreads) in enumerate(threads)
    for (ind2, time) in enumerate(times)
        for (ind1, seed) in enumerate(seeds)
            t, nd, nc = run_gibbs(
                max_time=time, 
                seed=seed,
                event_rate=event_rate,
                event_amplitude_mean=event_amplitude,
                bkgd_amplitude_mean=bkgd_amplitude,
                nthreads=nthreads
            )
            
            runtimes[ind1, ind2, ind3] = t
            data_sizes[ind1, ind2, ind3] = nd
            cluster_sizes[ind1, ind2, ind3] = nc

            @show (time, seed, t, nd, nc)
            println("\n")
        end
    end
end

results_path = (@__DIR__) * "/results.jld"
@save results_path runtimes data_sizes cluster_sizes
