import PointProcessSequences: logstudent_t_pdf

using Distributions: logpdf, MultivariateNormal
using Test: @test

function test_logstudent_t_pdf_limit()
    μ = zeros(2)
    Σ = [1.0 0.0; 0.0 1.0]
    x = [1.0; 2.0]
    ν = 1e10

    p1 = logpdf(MultivariateNormal(μ, Σ), x)
    p2 = logstudent_t_pdf(μ, Σ, ν, x)

    @show p1
    @show p2

    @assert isapprox(p1, p2; atol=1e-2)

    return true
end

@test test_logstudent_t_pdf_limit()