# ===
# Custom distributions
# ===


# ===
# GAMMA (rate parameterized)
# ===


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

posterior(volume::Real, n::Real, prior::RateGamma) = RateGamma(prior.α + n, prior.β + volume)

posterior(count_var::Real, prior::RateGamma) = RateGamma(prior.α + count_var, prior.β + 1)

Distributions.logpdf(g::RateGamma, x::Float64) = logpdf(Distributions.Gamma(g.α, 1 / g.β), x)

Distributions.rand(rng::AbstractRNG, g::RateGamma) = rand(rng, Distributions.Gamma(g.α, 1 / g.β))

Distributions.mean(g::RateGamma) = g.α / g.β

Distributions.var(g::RateGamma) = g.α / (g.β * g.β)

"""Specifies RateGamma distribution by first two moments."""
function specify_gamma(_mean::Real, _var::Real)
    β = _mean / _var
    α = _mean * β
    return RateGamma(α, β)
end




# ===
# INVERSE GAMMA
# ===


struct InverseGamma
    α::Float64
    β::Float64
end

Distributions.logpdf(g::InverseGamma, x) = logpdf(RateGamma(g.α, g.β), x)

Distributions.rand(rng::AbstractRNG, g::InverseGamma) = 1 / rand(rng, RateGamma(g.α, g.β))

Distributions.mean(g::InverseGamma) = (g.α > 1) ? g.β / (g.α - 1) : error("α must be > 1")

Distributions.var(g::InverseGamma) = 
    (g.α > 2) ? (g.β^2) / ((g.α - 1)^2 * (g.α - 2)) : error("α must be > 2")

function posterior(x̄, σ̄2, μ0, n, ν, prior::InverseGamma)
    α = prior.α + (n/2)
    β = prior.β + (1/2)*(n*σ̄2) + (1/2)*((n*ν)/(ν + n))*((x̄ - μ0)^2)
    return InverseGamma(α, β)
end

"""Specifies InverseGamma distribution by first two moments."""
function specify_inverse_gamma(μ::Real, σ²::Real)
    α = μ^2 / σ² + 2
    β = μ * (α - 1)
    return InverseGamma(α, β)
end




# ===
# SYMMETRIC DIRICHLET
# ===

"""
Symmetric Dirichlet distribution
"""
struct SymmetricDirichlet <: Distributions.ContinuousMultivariateDistribution
    conc::Float64
    dim::Int64
    log_normalizer::Float64
end
function SymmetricDirichlet(conc::Float64, dim::Int64)
    log_normalizer = (dim * lgamma(conc)) - lgamma(dim * conc)
    return SymmetricDirichlet(conc, dim, log_normalizer)
end

posterior(counts::AbstractVector, prior::SymmetricDirichlet) = Dirichlet(counts .+ prior.conc)

Distributions.length(g::SymmetricDirichlet) = g.dim

function Distributions._rand!(
    r::AbstractRNG, 
    g::SymmetricDirichlet, 
    x::AbstractVector{Float64}
)
    rand!(Distributions.Gamma(g.conc, 1.0), x)
    x ./= sum(x)
    return x
end

function Distributions.logpdf(g::SymmetricDirichlet, p::AbstractVector)
    return mapreduce(log, +, p) * (g.conc - 1) - g.log_normalizer
end




# ===
# SCALED INVERSE CHI SQUARED
# ===

struct ScaledInvChisq
    ν::Float64
    s2::Float64
end

function Distributions.rand(rng::AbstractRNG, g::ScaledInvChisq) 
    return g.ν * g.s2 / rand(rng, Chisq(g.ν))
end

function Distributions.logpdf(g::ScaledInvChisq, θ::Float64)
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




# === 
# NORMAL INVERSE CHI SQUARED
# ===

"""
A normal-inverse-chi-squared distribution is a convienent
prior on the mean and variance parameters for a univariate
Gaussian. It can be seen as a reparameterization of the
normal inverse gamma distribution. If:

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

function Distributions.rand(rng::AbstractRNG, g::NormalInvChisq)
    σ2 = rand(rng, ScaledInvChisq(g.ν, g.s2))
    μ = g.m + randn(rng) * sqrt(σ2 / g.k)
    return μ, σ2
end

function posterior(n::Int64, Σ_x::Float64, Σ_x²::Float64, prior::NormalInvChisq,)
    (n == 0) && return prior

    k = prior.k + n
    ν = prior.ν + n
    m = (Σ_x / k) + (prior.k * prior.m / k)
    s2 = (
        prior.ν * prior.s2
        + (Σ_x² - Σ_x * Σ_x / n)  # == sum_i (xᵢ - xbar)^2 
        + (prior.k * n * (Σ_x / n - prior.m)^2) / k
    ) / ν

    return NormalInvChisq(k, m, ν, s2)
end

function Distributions.logpdf(g::NormalInvChisq, μ::Float64, σ2::Float64)
    lp = logpdf(ScaledInvChisq(g.ν, g.s2), σ2)
    return lp + logpdf(Normal(g.m, sqrt(σ2 / g.k)), μ)
end




# ===
# SPARSE SAMPLE FROM MULTINOMIAL
# ===

function Distributions._logpdf(d::Multinomial, x::SparseVector{Int, Int})
    p = Distributions.probs(d)

    # Compute normalizer
    logp = logfactorial(Distributions.ntrials(d))

    # Compute unnormalized density
    for (i, xi) in zip(x.nzind, x.nzval)
        @inbounds logp += xi*log(p[i]) - logfactorial(xi)
    end

    return logp
end




# ===
# SYMMETRIC DIRICHLET MULTINOMIAL
# ===

struct SymmetricDirichletMultinomial
    conc::Real
    dim::Int
    n::Int
    log_normalizer::Real
end

function SymmetricDirichletMultinomial(conc::Real, dim::Int, n::Int)
    log_normalizer = logfactorial(n) + lgamma(dim * conc) - lgamma(n + dim * conc)
    return SymmetricDirichletMultinomial(conc, dim, n, log_normalizer)
end

Distributions.length(g::SymmetricDirichletMultinomial) = g.dim

function Distributions.logpdf(g::SymmetricDirichletMultinomial, x::SparseVector{Int64, Int64})
    # Log normalizer
    lp = g.log_normalizer

    # Likelihood
    lgamma_αk = lgamma(g.conc)
    @inbounds for xi in x.nzval
        lp += lgamma(xi + g.conc)
        lp -= logfactorial(xi) 
        lp -= lgamma_αk
    end
    
    return lp
end




# ===
# SPARSE DIRICHLET MULTINOMIAL
# ===

struct SparseDirichletMultinomial
    n::Int
    dense_conc::Real
    sparse_conc::SparseVector{Int, Int}
end

function Distributions.logpdf(d::SparseDirichletMultinomial, x::SparseVector{Int, Int})
    dim = length(d.sparse_conc)
    Σαₖ = dim * d.dense_conc + sum(d.sparse_conc)

    # Compute log-normalizer
    logp = logfactorial(d.n)
    logp += lgamma(Σαₖ)
    logp -= lgamma(d.n + Σαₖ)

    # Compute log of unnormalized density---if xᵢ = 0, the summand is zero, so we only 
    # need to iterate across the non-zero xᵢ
    for ind in 1:length(x.nzind)
        @inbounds i = x.nzind[ind]::Int
        @inbounds xᵢ = x.nzval[ind]::Int

        # PFLAG
        logp += lgamma(xᵢ + d.sparse_conc[i] + d.dense_conc)
        logp -= logfactorial(xᵢ) 
        logp -= lgamma(d.sparse_conc[i] + d.dense_conc)
    end
    
    return logp
end
