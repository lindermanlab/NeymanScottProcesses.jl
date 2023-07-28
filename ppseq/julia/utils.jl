"""
Draw one sample from a categorical distribution given the
(potentially unnormalized) log probabilities.
"""
sample_logprobs(lgp::Vector{Float64}) = rand(ds.Categorical(softmax(lgp)))


"""Creates identity matrix."""
eye(n::Int) = Matrix{Float64}(I, n, n)


"""Log-gamma function."""
lgamma(x) = logabsgamma(x)[1]
   

"""Gamma log pdf"""
function gammalogpdf(α::Float64, β::Float64, x::Float64)
    return α * log(β) - lgamma(α) + (α - 1) * log(x) - β * x
end

"""Logistic function"""
lgc(x) = 1 / (1 + exp(-x))
