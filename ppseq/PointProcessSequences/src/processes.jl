struct NSP end
struct MFM
    logV::Vector
end
struct DPM
    γ
end

"""
Construct a MFM with `K = rand(D)` clusters. The number of cluster is at most `K_max`.
"""
function MFM(K_max::Int, D)
    logV = zeros(K_max)

    # TODO

    return MFM(logV)
end

bkgd_prob(::NSP, model::AbstractModel) = model.bkgd_log_prob
bkgd_prob(::MFM, model::AbstractModel) = -Inf
bkgd_prob(::DPM, model::AbstractModel) = -Inf

new_cluster_prob(H::NSP, model::AbstractModel) = model.new_cluster_log_prob
new_cluster_prob(H::DPM, model::AbstractModel) = H.γ
function new_cluster_prob(H::MFM, model::AbstractModel)
    α = event_amplitude(model.priors).α
    k = num_events(model)
    return log(α) + H.logV[k+2] - H.logV[k+1]
end

function cluster_prob(::Union{NSP, MFM}, model::AbstractModel, event::AbstractEvent)
    α = event_amplitude(model.priors).α
    return log(datapoint_count(event) + α)
end
function cluster_prob(::DPM, model::AbstractModel, event::AbstractEvent)
    return log(datapoint_count(event))
end



PPSeq = FiniteMixture{
    BackgroundCluster{Spike},
    LowRankBackgroundCluster{Spike}, 
    MixtureFiniteMixture{SeqEvents}
}

SeqEvent{Spike} = Union{Uniform, Discrete, Dirichlet}

