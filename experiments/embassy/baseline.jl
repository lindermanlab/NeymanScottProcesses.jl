Base.@kwdef mutable struct BinModelDistribution
    E::Int
    V::Int
    T::Int
    K::Int = 1
    
    α::Real
    β::Real
    η::Real = 1.0
    γ::Real = 1.0
    α0::Real
    β0::Real
    η0::Real = 1.0
    γ0::Real = 1.0
    
    intervals
end

mutable struct BinModel
    A  # (T, K)
    ϵ  # (E, T, K)
    ω  # (V, T, K)
    A0  # scalar
    ϵ0  # (E,)
    ω0  # (V, E)
    intervals
end


get_dims(M::BinModelDistribution) = M.E, M.V, M.T, M.K

function get_dims(M::BinModel)
    T, K = size(M.A)
    E, V = size(M.ϵ, 1), size(M.ω, 1)
    
    return E, V, T, K
end

function num_clusters(M::BinModel)
    return size(M.A, 2)
end

function Base.rand(rng::AbstractRNG, P::BinModelDistribution)
    E, V, T, K = get_dims(P)
    
    A = rand(rng, Gamma(P.α, 1/P.β), T, K)
    
    ϵ = zeros(E, T, K)
    for (t, k) in product(1:T, 1:K)
        ϵ[:, t, k] = rand(rng, Dirichlet(E, P.η))
    end
    
    ω = zeros(V, T, K)
    for (t, k) in product(1:T, 1:K)
        ω[:, t, k] = rand(rng, Dirichlet(V, P.γ))
    end
    
    A0 = rand(rng, RateGamma(P.α0, P.β0))
    
    ϵ0 = rand(rng, Dirichlet(E, P.η0))
    
    ω0 = zeros(V, E)
    for e in 1:E
        ω0[:, e] = rand(rng, Dirichlet(V, P.γ0))
    end
    
    return BinModel(A, ϵ, ω, A0, ϵ0, ω0, P.intervals)
end




# ===
# Likelihoods
# ===

function Distributions.logpdf(P::BinModelDistribution, M::BinModel)
    E, V, T, K = get_dims(P)
        
    lp = 0.0
    
    D_A = Gamma(P.α, 1/P.β)
    D_ϵ = Dirichlet(E, P.η)
    D_ω = Dirichlet(V, P.γ)
    for (τ, k) in product(1:T, 1:K)
        lp += logpdf(D_A, M.A[τ, k])
        lp += logpdf(D_ϵ, M.ϵ[:, τ, k])
        lp += logpdf(D_ω, M.ω[:, τ, k])
    end
    
    lp += logpdf(RateGamma(P.α0, P.β0), M.A0)
    lp += logpdf(Dirichlet(E, P.η0), M.ϵ0)
    
    D_ω0 = Dirichlet(V, P.γ0)
    for e in 1:E
        lp += logpdf(D_ω0, M.ω0[:, e])
    end
    
    return lp
end

function timebin(t, intervals)
    valid_intervals = findall(τ -> (τ.start <= t < τ.stop), intervals)
    return valid_intervals[findfirst(τ -> τ != 0, valid_intervals)]
end

function log_intensity(x, k, θ::BinModel)
    if k == 0
        
        τ_length = (θ.intervals[0].stop - θ.intervals[0].start)
        amplitude = θ.A0
        embassy_log_prob = log(θ.ϵ0[x.e])
        word_log_prob = logpdf(Multinomial(sum(x.w), θ.ω0[:, x.e]), x.w)
    
    else  # k ∈ 1..K
        
        τ = timebin(x.t, θ.intervals) 
        
        τ_length = (θ.intervals[τ].stop - θ.intervals[τ].start)
        amplitude = θ.A[τ, k]
        embassy_log_prob = log(θ.ϵ[x.e, τ, k])
        word_log_prob = logpdf(Multinomial(sum(x.w), θ.ω[:, τ, k]), x.w)
        
    end
    
    return log(amplitude) + log(1/τ_length) + embassy_log_prob + word_log_prob
end

function log_intensity(x, model::BinModel)
    K = num_clusters(model)
    return logsumexp([log_intensity(x, k, model) for k in 0:K])
end

function integrated_intensity(model::BinModel)
    return model.A0 + sum(model.A)
end

function integral(M::BinModel, masks::Vector; num_samples=1000)
    # Compute intensity of the model in the masked region
    E, V, T, K = get_dims(M)
    
    # Add background element
    ∫λx = M.A0 * sum(volume.(masks)) / (M.intervals[0].stop - M.intervals[0].start)
    
    for (τ, k) in product(1:T, 1:K)
        
        num_in_region = 0
        for _ in 1:num_samples
            t = rand(Uniform(M.intervals[τ].start, M.intervals[τ].stop))
            e = rand(Categorical(M.ϵ[:, τ, k]))
            
            if Cable(t, e, spzeros(V)) in masks
                num_in_region += 1
            end
        end
                  
        ∫λx += M.A[τ, k] * (num_in_region / num_samples)
    end
    
    return ∫λx
end

function log_likelihood(data, model::BinModel)    
    return sum(x -> log_intensity(x, model), data) - integrated_intensity(model)
end

function log_likelihood(data, model::BinModel, masks::Vector)
    return log_likelihood(data, model) + integral(model, masks)
end

function log_joint(data, model::BinModel, prior::BinModelDistribution)
    return logpdf(prior, model) + log_likelihood(data, model)
end

function log_joint(data, model::BinModel, prior::BinModelDistribution, masks::Vector)
    return logpdf(prior, model) + log_likelihood(data, model, masks)
end




# ===
# Sampling
# ===

function parent_log_probs!(lp, x, model::BinModel)
    K = num_clusters(model)
    
    for k in 0:K
        lp[1+k] = log_intensity(x, k, model)
    end
    
    return lp
end

function gibbs_update_parents!(parents, data, model::BinModel)
    K = num_clusters(model)
    
    lp = zeros(K+1)
    for i in eachindex(parents)
        lp = parent_log_probs!(lp, data[i], model)        
        parents[i] = sample_logprobs!(lp) - 1
    end
    
    return parents
end

function compute_sufficient_statistics(data, parents, M::BinModel)
    E, V, T, K = get_dims(M)
    
    cluster_counts = zeros(T, K)
    embassy_counts = zeros(E, T, K)
    word_counts = zeros(V, T, K)
    bkgd_count = 0
    bkgd_embassy_counts = zeros(E)
    bkgd_word_counts = zeros(V, E)
    
    for (z, x) in zip(parents, data)
        if z == 0
            bkgd_count += 1
            bkgd_embassy_counts[x.e] += 1
            bkgd_word_counts[:, x.e] .+= x.w
        else
            τ = timebin(x.t, M.intervals) 
            cluster_counts[τ, z] += 1
            embassy_counts[x.e] += 1
            word_counts[:, τ, z] .+= x.w
        end
    end
    
    return cluster_counts, embassy_counts, word_counts, bkgd_count, bkgd_embassy_counts, bkgd_word_counts
end

function gibbs_update_model!(M::BinModel, P::BinModelDistribution, data, parents)
    E, V, T, K = get_dims(M)
    
    cluster_counts, embassy_counts, word_counts, bkgd_count, bkgd_embassy_counts, bkgd_word_counts =
        compute_sufficient_statistics(data, parents, M)
    
    # Clusters
    for (τ, k) in product(1:T, 1:K)
        # Amplitudes
        M.A[τ, k] = rand(RateGamma(P.α + cluster_counts[τ, k], P.β + 1))
        
        # Embassy distributions
        M.ϵ[:, τ, k] .= rand(Dirichlet(P.η .+ embassy_counts[:, τ, k]))
        
        # Word distributions
        M.ω[:, τ, k] .= rand(Dirichlet(P.γ .+ word_counts[:, τ, k]))
    end
    
    # Background amplitude
    M.A0 = rand(RateGamma(P.α0 + bkgd_count, P.β0 + 1))
    
    # Background embassy distributions
    M.ϵ0 .= rand(Dirichlet(P.η0 .+ bkgd_embassy_counts))
    
    # Background word distributions
    for e in 1:E
        M.ω0[:, e] .= rand(Dirichlet(P.γ0 .+ bkgd_word_counts[:, e]))
    end
    
    return M
end

get_timebin(t) = max(1, ceil(Int, t / bin_size))
