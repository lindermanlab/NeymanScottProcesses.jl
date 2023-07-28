"""
Cross product of any two priors. There is probably
a slick way to extend this to a cross product of
three or more distributions -- we can do that if
it ends up being useful.
"""
struct CrossPrior <: AbstractPrior
    priors::Vector{AbstractPrior}
end


log_normalizer(p::CrossPrior) =
    sum(log_normalizer(q) for q in p.priors)
