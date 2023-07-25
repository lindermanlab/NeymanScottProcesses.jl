# === Misc Utility Functions === #
const LOG_2_PI = log(2 * pi)

"""Log-gamma function."""
lgamma(x) = logabsgamma(x)[1]

"""
Sample from a list of log probabilities. Overwrites
vector so that no new memory is allocated.
"""
sample_logprobs!(log_probs) = sample(pweights(softmax!(log_probs)))


"""
Log normalizer of 1D normal distribution in information form.
given potential `m` and precision `v`. 

N(x | m, v) = exp{ m⋅x - .5v⋅x² - log Z(m, v) }
"""
gauss_info_logZ(m, v) = 0.5 * (LOG_2_PI - log(v) + m * m / v)

# GAMMA_MAX = 10_000
# GAMMA_X_ARR = 0 : 1 : GAMMA_MAX
# GAMMA_Y_ARR = lgamma.(GAMMA_RANGE)

# function fast_lgamma(x)
#     @inbounds return GAMMA_Y_ARR[min(floor(Int, x) + 1, GAMMA_MAX)]
# end

const LGAMMA_X = 50.0
const LGAMMA_0 = lgamma(LGAMMA_X)
const LGAMMA_1 = SpecialFunctions.digamma(LGAMMA_X)
const LGAMMA_2 = SpecialFunctions.trigamma(LGAMMA_X)

function taylor_lgamma(x::Float64)
    return LGAMMA_0 + LGAMMA_1*(x - LGAMMA_X) + LGAMMA_2*(x - LGAMMA_X)^2
end