# ================================ #
# ==== Dirichlet Distribution ==== #
# ================================ #
struct DirichletPrior <: AbstractPrior
    alpha::Vector{Float64}
    _log_normalizer::Float64
end

"""
Creates a Dirichlet distribution prior, with
pre-computed and log normalization constant.
"""
function DirichletPrior(alpha::Vector{Float64})
    Z = sum(lgamma.(alpha)) - lgamma(sum(alpha))
    return DirichletPrior(alpha, -Z)
end

log_normalizer(d::DirichletPrior) = d._log_normalizer
