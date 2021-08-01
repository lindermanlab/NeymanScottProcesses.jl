include("utils.jl")

f = joinpath(datadir, results_file)
results_jld = JLD.load(f)

# scores = [
#     mean([results_arr[1][j].results.test_log_p[end] for i in 1:num_trials])
#     for j = 1:num_parameters
# ]

# plot_kwargs = (xlabel="K / K_0", ylabel="test log like", legend=false, lw=6)
# plt = plot(θ_arr, scores, xlabel="K / K0", ylabel="test log like", legend=false, lw=6)