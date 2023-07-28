# ============================================= #
# ==== Normal Inverse Wishart Distribution ==== #
# ============================================= #

"""
Holds parameters of Normal-Inverse-Wishart distribution.

m (vector) : Mean.
m_n (scalar) : Number of pseudo-observations associated with m.
S (matrix) : Unnormalized covariance.
S_n (scalar) : Number of pseudo-observations associated with S.
_log_normalizer (scalar) : Log-normalization constant.
"""
struct NormInvWishartPrior <: AbstractPrior
    m::Vector{Float64}
    m_n::Float64
    S::Matrix{Float64}
    S_n::Float64
    _log_normalizer::Float64
end


"""
This constructor computes and caches the Log-normalization
constant up front, which results in a substantial speedup.
"""
function NormInvWishartPrior(
        m::Vector{Float64},
        m_n::Float64,
        S::Matrix{Float64},
        S_n::Float64
    )

    # Compute Log-normalizer.
    d = length(m)
    Z = -0.5 * (d * log(m_n) + S_n * logdet(S))
    for i = 1:d
        Z += lgamma(0.5 * (S_n + 1 - i))
    end

    # We omit the following constant term in Z:
    #   0.5 * (v + 1) * d * log(2) + 0.25 * d * (d + 1) * log(pi)

    return NormInvWishartPrior(m, m_n, S, S_n, Z)
end


# This cacheing trick cuts runtime down by ~30% !!
log_normalizer(niw::NormInvWishartPrior) = niw._log_normalizer
