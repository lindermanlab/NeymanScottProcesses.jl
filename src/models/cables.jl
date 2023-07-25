"""
Cables Neyman Scott Model 
Generative Model
================
Globals: 
    λ_0 ~ Gamma(α_0, β_0)
    K ~ Poisson(λ)
    K_0 ~ Poisson(λ_0)

Events:
    For k = 1, ..., K
        A_k ∼ Gamma(α, β)
        ϵ_k ~ Dirichlet(a_N)
        ν_k ~ Dirichlet(a_M)
        τ_k ~ Uniform([0,T])
        σ_k ~ InverseChisq

Background Datapoints:
    For i = 1, ..., K_0
        e_i ~ Categorical(ϵ_0)
        t_i ~ Uniform([0,T])
        v_i ~ Multinomial(ν_k0)

Event Datapoints:
    For k = 1, ..., K
        For i = 1, ..., Poisson(A_k) 
            e_i ~ Categorical(ϵ_k)
            t_i ~ Gaussian(τ_k, σ_k)
            v_i ~ Multinomial(ν_k)
"""

SAMPLED_CABLE_SIZE = 200
DAYS_PER_WEEK = 7  # Just in case I forget :)



# ===
# TYPES
# ===


struct Cable <: AbstractDatapoint{1}
    position::Float64
    embassy::Int
    words::SparseVector{Int, Int}
    _word_sum::Int
end

mutable struct CableCluster <: AbstractCluster{1}
    datapoint_count::Int
    first_moment::Float64
    second_moment::Float64
    embassy_count::Vector{Int}
    word_count::SparseVector{Int, Int}

    sampled_amplitude::Float64
    sampled_position::Float64
    sampled_variance::Float64
    sampled_embassy_probs::Vector{Float64}
    sampled_word_probs::Vector{Float64}
end

mutable struct CablesGlobals <: AbstractGlobals
    bkgd_rate::Float64
    bkgd_embassy_probs::Vector{Float64}
    bkgd_word_probs::Matrix{Float64}  # words x embassies
    day_of_week_probs::Vector{Float64}

    _bkgd_embassy_counts::Vector{Int}
    _bkgd_word_counts::Matrix{Int}  # words x embassies
    _day_of_week_counts::Vector{Int}
end

mutable struct CablesPriors <: AbstractPriors
    cluster_rate::Float64
    cluster_amplitude::RateGamma
    variance_amplitude::InverseGamma
    embassy_concentration::Vector{Float64}
    word_concentration::Float64

    bkgd_amplitude::RateGamma
    bkgd_embassy_concentration::Vector{Float64}
    bkgd_word_concentration::Matrix{Float64}
    day_of_week::Vector{Float64}
end

struct CablesMask <: AbstractMask
    t1::Real
    t2::Real
    embassy::Int
    embassy_dim::Int
end

struct CablesComplementMask <: AbstractMask
    T::Real
    masks::Vector{CablesMask}
end

const CablesModel = NeymanScottModel{1, Cable, CableCluster, CablesGlobals, CablesPriors}




# ===
# UTILITY METHODS
# ===

_sample_num_words() = rand(Poisson(SAMPLED_CABLE_SIZE))

trim(x, lower, upper) = max(min(x, upper), lower)

num_words(c::Cable) = c._word_sum

get_day_of_week(i) = 1 + (floor(Int, i) % DAYS_PER_WEEK)

get_day_of_week(c::Cable) = get_day_of_week(position(c))

time(c::Cable) = c.position

word_dim(priors::CablesPriors) = size(priors.bkgd_word_concentration, 1)

word_dim(m::CablesModel) = word_dim(m.priors)

embassy_dim(priors::CablesPriors) = length(priors.embassy_concentration)

embassy_dim(m::CablesModel) = embassy_dim(m.priors)

function logpdf_tdist(μ, σ, ν, x)
    # Normalize to a standard (zero mean, unit variance) t-distribution
    z = (x - μ) / σ
    return logpdf(TDist(ν), z)
end




# ===
# CONSTRUCTORS
# ===


Cable(position, embassy, words) = Cable(position, embassy, words, sum(words))

constructor_args(c::CableCluster) = (length(c.embassy_count), length(c.word_count))

function CableCluster(A, μ, σ2, ϵ, ν)
    ϵ_count = zeros(length(ϵ))
    ν_count = spzeros(length(ν))
    return CableCluster(0, 0.0, 0.0, ϵ_count, ν_count, A, μ, σ2, ϵ, ν)   
end

function CableCluster(num_embassies::Int, num_words::Int)
    ϵ = zeros(num_embassies)
    ν = zeros(num_words)
    return CableCluster(NOT_SAMPLED_AMPLITUDE, 0.0, 0.0, ϵ, ν)   
end

function CablesGlobals(bkgd_rate, bkgd_embassy_probs, bkgd_word_probs, day_of_week_probs)
    return CablesGlobals(
        bkgd_rate,
        bkgd_embassy_probs,
        bkgd_word_probs,
        day_of_week_probs,
        zeros(Int, size(bkgd_embassy_probs)),
        zeros(Int, size(bkgd_word_probs)),
        zeros(Int, DAYS_PER_WEEK),
    )
end

function CablesModel(max_time::Float64, priors::CablesPriors; max_radius::Float64=Inf)
    globals = sample(priors)
    clusters = ClusterList(CableCluster(embassy_dim(priors), word_dim(priors)))

    model = CablesModel(
        max_time,
        max_radius,
        priors,
        globals,
        clusters,
        0.0,
        0.0,
        Float64[],
        Dict(
            :word_buffer => ones(word_dim(priors)),
            :embassy_buffer => ones(embassy_dim(priors)),
        )
    )
    
    # Compute the background and new cluster probabilities.
    _reset_model_probs!(model)

    return model
end




# ===
# DATA MANAGEMENT
# ===


function reset!(cluster::CableCluster)
    cluster.datapoint_count = 0.0
    cluster.embassy_count .= 0.0
    cluster.word_count .= 0.0
    cluster.first_moment = 0.0
    cluster.second_moment = 0.0
end

function remove_datapoint!(
    model::CablesModel, 
    x::Cable, 
    k::Int64;
    recompute_posterior::Bool=true
) 
    e = clusters(model)[k]

    # If this is the last spike in the event, we can return early.
    (e.datapoint_count == 1) && (return remove_cluster!(clusters(model), k))

    # Otherwise, update sufficient statistics
    e.datapoint_count -= 1
    e.first_moment -= time(x)
    e.second_moment -= time(x)^2
    e.embassy_count[x.embassy] -= 1
    @. e.word_count -= x.words

    # Recompute posterior based on new sufficient statistics
    recompute_posterior && set_posterior!(model, k)

    return k
end

function add_datapoint!(
    model::CablesModel, 
    x::Cable, 
    k::Int64;
    recompute_posterior::Bool=true
) 
    e = clusters(model)[k]

    # Update sufficient statistics
    e.datapoint_count += 1
    e.first_moment += time(x)
    e.second_moment += time(x)^2
    e.embassy_count[x.embassy] += 1
    @. e.word_count += x.words

    # Recompute posterior based on new sufficient statistics
    recompute_posterior && set_posterior!(model, k)

    return k
end



# ===
# PROBABILITIES
# ===

function log_bkgd_intensity(model::CablesModel, x::Cable)
    # Background rate
    lp = log(bkgd_rate(model.globals))

    # Embassy contribution
    lp += log(model.globals.bkgd_embassy_probs[x.embassy])

    # Word contribution
    D_ν = Multinomial(num_words(x), model.globals.bkgd_word_probs[:, x.embassy])
    lp += logpdf(D_ν, x.words)

    # Day of week contribution
    lp += log(DAYS_PER_WEEK * model.globals.day_of_week_probs[get_day_of_week(x)])

    return lp
end

function log_cluster_intensity(model::CablesModel, e::CableCluster, x::Cable)
    # Amplitude
    lp = log(amplitude(e))

    # Time contribution
    lp += logpdf(Normal(e.sampled_position, e.sampled_variance), position(x))

    # Embassy contribution
    lp += logpdf(Categorical(e.sampled_embassy_probs), x.embassy)

    # Word contribution
    lp += logpdf(Multinomial(num_words(x), e.sampled_word_probs), x.words)

    return lp 
end

function log_p_latents(model::CablesModel)
    priors = model.priors
    globals = model.globals

    lp = 0.0
    for event in events(model)
        # Log prior of event amplitude
        lp += logpdf(cluster_amplitude(priors), event.sampled_position)

        # Log prior of event variance
        lp += logpdf(priors.variance_amplitude, event.sampled_variance)

        # Log prior of event embassy probabilities
        lp += logpdf(Dirichlet(priors.embassy_concentration), event.sampled_embassy_probs)

        # Log prior of event word probabilities
        # TODO Optimize performance
        # D_ν = model.buffers[:word_dirichlet]
        D_ν = Dirichlet(word_dim(model), priors.word_concentration)
        lp += logpdf(D_ν, event.sampled_word_probs)        
    end

    # Log prior on position
    lp -= log(volume(model)) * length(events(model))
end

function log_prior(model::CablesModel)
    priors = model.priors
    globals = model.globals

    lp = 0.0

    # Background rate
    lp += logpdf(priors.bkgd_amplitude, globals.bkgd_rate)

    # Background embassy probabilities
    lp += logpdf(Dirichlet(priors.bkgd_embassy_probs), globals.bkgd_embassy_probs)

    # Background word probabilities
    D_ν = Dirichlet(model.buffers[:word_buffer])
    for embassy in 1:embassy_dim(model)
        D_ν.alpha .= view(priors.bkgd_word_concentration, :, embassy)
        lp += logpdf(D_ν, view(globals.bkgd_word_probs, :, embassy))
    end

    # Day of week probabilities
    lp += logpdf(priors.day_of_week, globals.day_of_week_probs)

    return lp
end

function bkgd_log_like(m::CablesModel, x::Cable)
    # Temporal contribution
    ll = -log(volume(m) * DAYS_PER_WEEK)

    # Embbasy contribution
    # TODO Optimize via caching
    ϵ_stat = m.buffers[:embassy_buffer]
    ϵ_stat .= m.priors.bkgd_embassy_concentration
    ϵ_stat .+= m.globals._bkgd_embassy_counts
    ϵ_stat ./= sum(ϵ_stat)

    ϵ_posterior = Categorical(ϵ_stat)
    ll += logpdf(ϵ_posterior, x.embassy)
    
    # Word count contribution
    # TODO Optimize via caching
    ν_stat = m.buffers[:word_buffer]
    ν_stat .= view(m.priors.bkgd_word_concentration, :, x.embassy)
    ν_stat .+= view(m.globals._bkgd_word_counts, :, x.embassy)

    ν_posterior = DirichletMultinomial(num_words(x), ν_stat)
    ll += logpdf(ν_posterior, x.words)

    # Day of week contribution
    pδ = m.priors.day_of_week + m.globals._day_of_week_counts
    δ_posterior = Categorical(pδ / sum(pδ))
    ll += logpdf(δ_posterior, get_day_of_week(x))

    return ll
end

function log_posterior_predictive(c::CableCluster, x::Cable, model::CablesModel)
    priors = model.priors

    # Temporal component
    # Prior parameters
    μ0 = 0
    ν0 = 0
    α0 = priors.variance_amplitude.α
    β0 = priors.variance_amplitude.β

    # Useful quantities
    Σxᵢ, Σxᵢ² = c.first_moment, c.second_moment
    n = datapoint_count(c)
    x̄ = Σxᵢ / n
    σ̄ = Σxᵢ² + n*x̄^2 - 2x̄*Σxᵢ 

    # Posterior statistics
    μn = (ν0*μ0 + n*x̄) / (ν0 + n)
    νn = ν0 + n
    αn = α0 + n/2
    βn = β0 + σ̄/2 + (n * ν0 * (x̄ - μ0)^2)/(2 * (ν0 + n))

    ll = logpdf_tdist(μn, βn*(νn+1)/(αn*νn), 2*αn, position(x))

    # Embbasy contribution
    # TODO Optimize via caching
    ϵ_stat = model.buffers[:embassy_buffer]
    ϵ_stat .= priors.embassy_concentration
    ϵ_stat .+= c.embassy_count
    ϵ_stat ./= (datapoint_count(c) + sum(priors.embassy_concentration))

    ll += logpdf(Categorical(ϵ_stat), x.embassy)

    # Word contribution
    # TODO Optimize ... via caching?
    ν_prior = priors.word_concentration
    ν_cluster = c.word_count
    ν_posterior = SparseDirichletMultinomial(num_words(x), ν_prior, ν_cluster)
    ll += logpdf(ν_posterior, x.words)

    return ll
end 

function log_posterior_predictive(x::Cable, model::CablesModel)
    priors = model.priors

    # Temporal contribution
    ll = -log(volume(model))

    # Embassy contribution
    # TODO Optimize via caching
    ϵ_stat = model.buffers[:embassy_buffer]
    ϵ_stat .= priors.embassy_concentration
    ϵ_stat ./= sum(ϵ_stat)
    ll += logpdf(Categorical(ϵ_stat), x.embassy)

    # Word contribution
    # TODO Optimize via caching
    ν_prior = priors.word_concentration
    ν_predictive = SymmetricDirichletMultinomial(ν_prior, word_dim(model), num_words(x))
    ll += logpdf(ν_predictive, x.words)

    return ll
end



# ===
# SAMPLING
# ===


function sample(priors::CablesPriors)
    # Sample background parameters
    bkgd_rate = rand(priors.bkgd_amplitude)
    bkgd_embassy_probs = rand(Dirichlet(priors.bkgd_embassy_concentration))

    # Sample background word probabilities
    bkgd_word_probs = zeros(size(priors.bkgd_word_concentration))
    for embassy in 1:embassy_dim(priors)
        bkgd_word_probs[:, embassy] .= rand(Dirichlet(priors.bkgd_word_concentration[:, embassy]))
    end
    
    # Sample day of week probabilities
    day_of_week_probs = rand(Dirichlet(Vector(priors.day_of_week)))

    return CablesGlobals(bkgd_rate, bkgd_embassy_probs, bkgd_word_probs, day_of_week_probs)
end

function sample_cluster(globals::CablesGlobals, model::CablesModel)
    priors = model.priors

    A = rand(cluster_amplitude(priors))
    μ = rand() * volume(model)
    σ2 = rand(priors.variance_amplitude)
    ϵ = rand(Dirichlet(priors.embassy_concentration))
    ν = rand(Dirichlet(word_dim(model), priors.word_concentration))

    return CableCluster(A, μ, σ2, ϵ, ν)
end

function sample_datapoint(globals::CablesGlobals, model::CablesModel)
    max_time = volume(model)
    num_weeks = floor(Int, max_time / DAYS_PER_WEEK)

    day = float(DAYS_PER_WEEK*rand(0:num_weeks) + rand(Categorical(globals.day_of_week_probs)))
    embassy = rand(Categorical(globals.bkgd_embassy_probs))
    words = rand(Multinomial(_sample_num_words(), globals.bkgd_word_probs[:, embassy]))

    return Cable(trim(day, 0, max_time), embassy, words)
end

function sample_datapoint(c::CableCluster, G::CablesGlobals, M::CablesModel)
    max_time = volume(M)

    day = rand(Normal(position(c), sqrt(c.sampled_variance)))
    embassy = rand(Categorical(c.sampled_embassy_probs))
    words = rand(Multinomial(_sample_num_words(), c.sampled_word_probs))

    return Cable(trim(day, 0, max_time), embassy, words)
end




# ===
# GIBBS SAMPLING
# ===


function gibbs_sample_cluster_params!(cluster::CableCluster, m::CablesModel)
    priors = m.priors

    # Sample amplitude
    n = datapoint_count(cluster)
    cluster.sampled_amplitude = rand(posterior(n, cluster_amplitude(priors)))

    # Sample mean and variance in time
    α, β = priors.variance_amplitude.α, priors.variance_amplitude.β
    h, J = cluster.first_moment, cluster.second_moment

    μ = h / n
    Ψ = J - h^2 / n
    σ2 = rand(InverseGamma(α + n/2, β + Ψ/2))

    cluster.sampled_position = μ + randn() * sqrt(σ2)
    cluster.sampled_variance = σ2

    # Sample embassy distribution
    ϵ_0, ϵ_count = priors.embassy_concentration, cluster.embassy_count
    cluster.sampled_embassy_probs = rand(Dirichlet(ϵ_0 + ϵ_count))

    # Sample word distribution
    ν_0, ν_count = priors.word_concentration, cluster.word_count
    ν = Dirichlet(m.buffers[:word_buffer])
    @. ν.alpha = ν_0
    for (i, xi) in zip(ν_count.nzind, ν_count.nzval)
        ν.alpha[i] += xi
    end

    cluster.sampled_word_probs = rand(ν)
end

function gibbs_sample_globals!(m::CablesModel, data::Vector{Cable}, assignments::Vector{Int})
    priors = m.priors
    globals = m.globals

    # Sample background rate
    num_bkgd_points = mapreduce(k->(k==-1), +, assignments)
    globals.bkgd_rate = rand(posterior(volume(m), num_bkgd_points, bkgd_amplitude(priors)))

    # Sample background embassy probabilities
    ϵ_0, ϵ_count = priors.bkgd_embassy_concentration, globals._bkgd_embassy_counts
    ϵ_posterior = Dirichlet(ϵ_0 + ϵ_count)
    globals.bkgd_embassy_probs .= rand(ϵ_posterior)

    # Sample background word probabilities
    ν_0, ν_count = priors.bkgd_word_concentration, globals._bkgd_word_counts
    ν_posterior = Dirichlet(ones(word_dim(m)))
    @inbounds for embassy in 1:embassy_dim(m)
        ν_posterior.alpha .= view(ν_0, :, embassy)
        ν_posterior.alpha .+= view(ν_count, :, embassy)
        rand!(ν_posterior, view(globals.bkgd_word_probs, :, embassy))
    end

    # Sample day of week probabilities
    δ_posterior = Dirichlet(priors.day_of_week + globals._day_of_week_counts)
    globals.day_of_week_probs .= rand(δ_posterior)
end

function add_bkgd_datapoint!(model::CablesModel, x::Cable)
    model.globals._bkgd_embassy_counts[x.embassy] += 1
    model.globals._bkgd_word_counts[:, x.embassy] .+= x.words
    model.globals._day_of_week_counts[get_day_of_week(x)] += 1
end

function remove_bkgd_datapoint!(model::CablesModel, x::Cable)
    model.globals._bkgd_embassy_counts[x.embassy] -= 1
    model.globals._bkgd_word_counts[:, x.embassy] .-= x.words
    model.globals._day_of_week_counts[get_day_of_week(x)] -= 1
end

function gibbs_initialize_globals!(model::CablesModel, data::Vector{Cable}, ω::Vector{Int})
    priors = model.priors
    globals = model.globals

    globals._bkgd_embassy_counts .= 0
    globals._bkgd_word_counts .= 0
    globals._day_of_week_counts .= 0
    
    @inbounds for i in 1:length(data)
        (ω[i] == -1) && add_bkgd_datapoint!(model, data[i])
    end
end




# ===
# MASKING
# ===

# Regular mask

Base.in(x::Cable, mask::CablesMask) = 
    (mask.t1 <= x.position < mask.t2) && (mask.embassy == x.embassy)

volume(mask::CablesMask) = 
    (mask.t2 - mask.t1) / mask.embassy_dim

complement_masks(masks::Vector{CablesMask}, model::CablesModel) = 
    CablesComplementMask[CablesComplementMask(volume(model), masks)]

function create_random_mask(
    model::CablesModel, 
    interval_length::Real, 
    pc_masked::Real
)
    T, E = volume(model), embassy_dim(model)
    τ = interval_length

    @assert T > τ

    # Fill model with disjoint masks
    potential_masks = CablesMask[
        CablesMask(t1, t1 + τ, e, E)
        for t1 in 1:τ:(T-τ), e in 1:E
    ]

    # Sample masks
    mask_volume = volume(first(potential_masks))
    num_masks = floor(Int, T * pc_masked / mask_volume)

    return sample(potential_masks, num_masks, replace=false)
end

# Complement mask

Base.in(x::Cable, comp_mask::CablesComplementMask) = !(x in comp_mask.masks)

volume(comp_mask::CablesComplementMask) = 
    comp_mask.T - sum(volume.(comp_mask.masks))