
# === IMPORTED FUNCTIONS AND TYPES === #

import Distributions.rand!
import Distributions._rand!
import Distributions.logpdf
import Distributions.pdf
import Distributions.ContinuousUnivariateDistribution
import Distributions.ContinuousMultivariateDistribution
import Distributions.DirichletMultinomial

# === IMPORTED DISTRIBUTIONS FROM DISTRIBUTIONS.JL === #

import Distributions.Dirichlet
import Distributions.Poisson
import Distributions.Chisq
import Distributions.Normal
import Distributions.Categorical
import Distributions.MultivariateNormal
import Distributions.InverseWishart
import Distributions.Multinomial
import Distributions.TDist

# === HELPER FUNCTIONS TO SET HYPERPARAMS === #

"""Specifies RateGamma distribution by first two moments."""
function specify_gamma(_mean::Real, _var::Real)
    β = _mean / _var
    α = _mean * β
    RateGamma(α, β)
end


# === GAMMA DISTRIBUTION === #

"""
RateGamma distribution, parameterized by shape and rate.

p(x | α, β) = (β^α / Γ(α)) ⋅ x^(α + 1) ⋅ exp{-βx}

This is a light wrapper to Distributions.jl which
uses the alternative parameterization based on
shape and scale to characterize the gamma distribution.
"""
struct RateGamma
    α::Float64
    β::Float64
end

logpdf(g::RateGamma, x::Float64) = logpdf(Distributions.Gamma(g.α, 1 / g.β), x)
rand(g::RateGamma) = rand(Distributions.Gamma(g.α, 1 / g.β))
mean(g::RateGamma) = g.α / g.β
var(g::RateGamma) = g.α / (g.β * g.β)

posterior(count_var::Real, prior::RateGamma) = (
    RateGamma(prior.α + count_var, prior.β + 1)
)

# === SYMMETRIC DIRICHLET === #

struct SymmetricDirichlet <: ContinuousMultivariateDistribution
    conc::Float64
    dim::Int64
    log_normalizer::Float64
end

function SymmetricDirichlet(conc::Float64, dim::Int64)
    SymmetricDirichlet(
        conc,
        dim,
        (dim * lgamma(conc)) - lgamma(dim * conc)  # log normalizer.
    )
end

length(g::SymmetricDirichlet) = g.dim

function _rand!(
        r::Random.AbstractRNG,
        g::SymmetricDirichlet,
        x::AbstractVector{Float64}
    )

    rand!(Distributions.Gamma(g.conc, 1.0), x)
    x ./= sum(x)
end

posterior(counts::AbstractVector, prior::SymmetricDirichlet) = (
    Dirichlet(counts .+ prior.conc)
)

logpdf(g::SymmetricDirichlet, p::AbstractVector) = (
    mapreduce(log, +, p) * (g.conc - 1) - g.log_normalizer
)


# === SCALED INVERSE CHI SQUARED === #

struct ScaledInvChisq
    ν::Float64
    s2::Float64
end

function logpdf(g::ScaledInvChisq, θ::Float64)
    # See pg 576 of Gelman et al., Bayesian Data Analysis.
    hν = 0.5 * g.ν
    return (
        hν * log(hν)
        - lgamma(hν)
        + hν * log(g.s2)
        - (hν + 1) * log(θ)
        - hν * g.s2 / θ
    )
end

rand(g::ScaledInvChisq) = g.ν * g.s2 / rand(Chisq(g.ν))

# === NORMAL INVERSE CHI SQUARED === #

"""
A normal-inverse-chi-squared distribution is a convienent
prior on the mean and variance parameters for a univariate
Gaussian. It can be seen as a reparameterization of the
normal inverse gamma distribution.

If:

    σ² ~ Inv-χ²(ν, s)
    μ | σ² ~ Normal(m, σ² / k)

Then (μ, σ²) follows a normal-inverse-chi-squared distribution
with parameters (k, m, ν, s)

Parameters
----------
k : Number of psuedo-observations of m. Note that k > 0.
m : Prior on the mean parameter.
ν : Number of psuedo-observations of s. Note that ν > 0.
s2 : Prior on the variance parameter. Note that s2 > 0.

References
----------
 - Pgs 42-43, 67-69 of Gelman et al., Bayesian Data Analysis
 - Pg 134 of Murphy, Machine Learning: A Probabilistic Perspective.
"""
struct NormalInvChisq
    k::Float64
    m::Float64
    ν::Float64
    s2::Float64
end

function rand(g::NormalInvChisq)
    σ2 = rand(ScaledInvChisq(g.ν, g.s2))
    μ = g.m + randn() * sqrt(σ2 / g.k)
    return μ, σ2
end

function posterior(
        n::Int64,
        sum_x::Float64,
        sumsq_x::Float64,
        prior::NormalInvChisq,
    )

    (n == 0) && return prior

    k = prior.k + n
    ν = prior.ν + n
    m = (sum_x / k) + (prior.k * prior.m / k)
    s2 = (
        prior.ν * prior.s2
        + (sumsq_x - sum_x * sum_x / n)  # == sum_i (xᵢ - xbar)^2 
        + (prior.k * n * (sum_x / n - prior.m)^2) / k
    ) / ν

    return NormalInvChisq(k, m, ν, s2)
end

function logpdf(g::NormalInvChisq, μ::Float64, σ2::Float64)
    return (
        logpdf(ScaledInvChisq(g.ν, g.s2), σ2)
        + logpdf(Normal(g.m, sqrt(σ2 / g.k)), μ)
    )
end



# === SparseDirichletMultinomial === #

"""
A sparse Dirichlet-Multinomial distribution implements a Dirichlet-Multinomial
distribution, but allows for a sparse component and a dense component.
"""
struct SparseDirichletMultinomial
    num_samples::Int
    offset::Float64
    sparse_part::SparseVector{Int, Int}
end


"""
Compute the log pmf of the Dirichlet-multinomial distribution `d` at the
sample `x`.

    p_d(x) = (
        log(n!) + logΓ(Σ d) - logΓ(n + Σ d)
        + Σᵢ logΓ(xᵢ + dᵢ) - log(xᵢ!) - logΓ(dᵢ)
    )
"""
function logpdf(
    d::SparseDirichletMultinomial, 
    x::SparseVector{Int, Int}, 
    lgamma_buffer::Vector{Float64}
)
    
    dim = length(d.sparse_part)
    dirichlet_sum = dim * d.offset + sum(d.sparse_part)

    # Compute log-normalizer
    logp = logfactorial(d.num_samples)
    logp += lgamma(dirichlet_sum)
    logp -= lgamma(d.num_samples + dirichlet_sum)

    # Compute log of unnormalized density
    # When xᵢ = 0, then the term in the summation cancels out
    # So we only need to iterate across the non-zero entries of x
    for ind in 1:length(x.nzind)
        @inbounds i = x.nzind[ind]::Int
        @inbounds xi = x.nzval[ind]::Int

        # PFLAG
        logp += lgamma(xi + d.sparse_part[i] + d.offset)
        #logp += lgamma_buffer[xi + d.sparse_part[i] + 1]

        logp -= logfactorial(xi) 

        logp -= lgamma(d.sparse_part[i] + d.offset)
        #logp -= lgamma_buffer[d.sparse_part[i] + 1]

    end
    
    return logp
end


# === SparseMultinomial === #

"""
A sparse multinomial distribution implements a multinomial
distribution, but allows for a sparse sample.
"""
struct SparseMultinomial
    num_samples::Int
    p::Vector{Float64}
end

function logpdf(d::SparseMultinomial, x::SparseVector{Int, Int})

    # Compute normalizer
    prob = logfactorial(d.num_samples)

    # Compute unnormalized density
    for (i, xi) in zip(x.nzind, x.nzval)
        @inbounds prob += xi*log(d.p[i]) - logfactorial(xi)
    end

    return prob
end

pdf(d::SparseMultinomial, x::SparseVector{Int, Int}) = exp(logpdf(d, x))
