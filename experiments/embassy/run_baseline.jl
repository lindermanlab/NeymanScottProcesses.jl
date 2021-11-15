include("util.jl")


# ===
# Configuration
# ===

# Local imports
using Base.Iterators: product
using StatsFuns: logsumexp
using NeymanScottProcesses: sample_logprobs!, complement_masks


# Local config parameters
NUM_BINS = 4
NUM_SAMPLES = 100



# Load external config
seed = CONFIG[:seed]
min_date = get_dateid(CONFIG[:min_date])
max_date = get_dateid(CONFIG[:max_date])
vocab_cutoff = CONFIG[:vocab_cutoff]




# ===
# Load data
# ===

# Load data
cables, embassy_dim, word_dim = 
     construct_cables(DATADIR, min_date, max_date, vocab_cutoff)

word_distr = load_empirical_word_distribution(DATADIR)
normalized_word_distr = word_distr ./ sum(word_distr, dims=1)

# Load NSP data
data, _, _, _, _, nsp_model = load_results(CONFIG, "nsp_fit.jld")

# Mask data
Random.seed!(12345)
masks = create_random_mask(nsp_model, 1.0, CONFIG[:percent_masked])
masked_data, unmasked_data = split_data_by_mask(data, masks)

meta = load_cables_metadata(DATADIR)

@show length(cables), length(data)
@show length(masked_data), length(unmasked_data)
@show embassy_dim word_dim
@show max_date - min_date;




# ===
# Make intervals, train, and test set
# ===

# Intervals
max_time = max_date - min_date + 0.1
bin_size = max_time / NUM_BINS

intervals = Dict(τ => (start=(τ-1)*bin_size, stop=τ*bin_size) for τ in 1:NUM_BINS)
intervals[0] = (start=0.0, stop=max_time)

# Train
t_arr = [c.position for c in unmasked_data]
e_arr = [c.embassy for c in unmasked_data]
w_arr = [c.words for c in unmasked_data]
x_arr = [(t=t, e=e, w=w) for (t, e, w) in zip(t_arr, e_arr, w_arr)]

# Test
t_arr_test = [c.position for c in masked_data]
e_arr_test = [c.embassy for c in masked_data]
w_arr_test = [c.words for c in masked_data]
x_arr_test = [
    (t=t, e=e, w=w) for (t, e, w) in 
    zip(t_arr_test, e_arr_test, w_arr_test)
]




# ===
# Set up model
# ===

# Data parameters
E = size(word_distr, 2)
V = length(w_arr[1])
T = length(intervals) - 1
K = 1

# Useful quantities
num_datapoints = length(cables)
bkgd_size = num_datapoints / 2
avg_cluster_size = (num_datapoints - bkgd_size) / (T * K)

# Construct prior
A_prior = specify_gamma(avg_cluster_size, avg_cluster_size^2)
A0_prior = specify_gamma(bkgd_size, bkgd_size^2)
prior = BinModelDistribution(
    E=E, V=V, T=T,
    intervals=intervals,
    K = K,
    
    α = A_prior.α,
    β = A_prior.β,
    η = 1.0,
    γ = 1.0,
    
    α0 = A0_prior.α,
    β0 = A0_prior.β,
    η0 = 1.0,
    γ0 = 1.0,
)





# ===
# Run inference
# ===

Random.seed!(11)

train_data = x_arr
test_data = x_arr_test

# Initialize
model = rand(MersenneTwister(1), prior)
parents = rand(0:K, length(x_arr))
comp_masks = complement_masks(masks, nsp_model)

@show integrated_intensity(model)
@show integral(model, masks)

# Track information
z_hist = [copy(parents)]
lp_hist = [log_joint(train_data, model, prior, masks)]
test_ll_hist = [log_likelihood(test_data, model, comp_masks)]

history = (z=z_hist, trainlp=lp_hist, testll=test_ll_hist)

println("Log probability: $(lp_hist[1]), Test loglike: $(test_ll_hist[1])")
@time for itr in 1:NUM_SAMPLES
    gibbs_update_parents!(parents, train_data, model)
    gibbs_update_model!(model, prior, train_data, parents)
    
    
    lp = log_joint(train_data, model, prior, masks)
    test_ll = log_likelihood(test_data, model, comp_masks)
    
    push!(history.z, copy(parents))
    push!(history.trainlp, lp)
    push!(history.testll, test_ll)
    
    println("Iteration $itr, Log probability: $lp, Test loglike: $test_ll")
end
println()



# ===
# Save results
# ===

savepath = joinpath(RESULTSDIR, "baseline_fit.jld")
JLD.@save savepath prior masks model parents history
