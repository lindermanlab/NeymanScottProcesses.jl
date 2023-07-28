# == Bunch of imports == #
using LinearAlgebra
using Statistics
using SparseArrays

import Distributions
import Random
import PyPlot

const ds = Distributions
const rnd = Random
const plt = PyPlot

# This is a hack to import the module for now, we can
# make PointProcessMixtures a full package later.
!(pwd() in LOAD_PATH) && push!(LOAD_PATH, pwd())
include("PointProcessMixtures.jl")
using Main.PointProcessMixtures


function nodetimemark_demo(
        n_clusters, n_embassies, n_words,
        events_per_cluster, n_gibbs_samples;
        method="dpmm", seed=1234
    )

    # Set random seed.
    rnd.seed!(seed)

    words_per_doc = 100

    # === MAKE SOME DATA === #

    node_prior = ds.Dirichlet(ones(n_embassies))
    time_prior = ds.Uniform(0, 10)
    mark_prior = ds.Dirichlet(ones(n_words))

    node_dists = []
    time_dists = []
    mark_dists = []
    data = []

    for k = 1:n_clusters

        # Create new latent event with distribution
        # over embassies, cable timestamps, and content.
        push!(
            node_dists,
            ds.Categorical(rand(node_prior))
        )
        push!(
            time_dists,
            ds.Normal(rand(time_prior), 1.0)
        )
        push!(
            mark_dists,
            ds.Multinomial(100, rand(mark_prior))
        )

        # Sample cable arising from latent event k.
        for i = 1:events_per_cluster

            _n = spzeros(Int64, n_embassies)
            _n[rand(node_dists[end])] = 1

            push!(
                data, (
                    _n,
                    rand(time_dists[end]),
                    SparseVector(rand(mark_dists[end])),
                )
            )
        end
    end

    # === SPECIFY PRIO === #

    # Prior on full latent event
    prior = CrossPrior([
        DirichletPrior(ones(n_embassies)),
        NormInvWishartPrior(
            [5.0], 0.001, [1.0][:, :], 1.001),
        DirichletPrior(ones(n_embassies))
    ])

    # Initial cluster assignments.
    num_init_clusters = 1
    _a, _c = initialize_clusters(
        data, num_init_clusters, prior, NodeTimeMarkCluster)

    # Specify clustering model.
    if method == "dpmm"

        alpha = 10.0
        model = DPMM(alpha, prior, _a, _c)

    elseif method == "mfmm"

        gamma = 1.0
        k_distrib = ds.Poisson(20.0)
        model = MFMM(gamma, k_distrib, prior, _a, _c)

    elseif method == "nspmm"

        min_datapoints = length(data) - 1
        max_datapoints = length(data) + 1
        max_clusters = 10 * n_clusters

        alpha_0 = 1.0
        beta_0 = log(max_clusters) - log(10)
        alpha = 1.0
        beta = 0.9

        upper_lim = [1.0, 1.0]
        lower_lim = [0.0, 0.0]

        model = NSPMM(
            min_datapoints,
            max_datapoints,
            max_clusters,
            alpha_0, beta_0, alpha, beta,
            upper_lim, lower_lim,
            prior, _a, _c)

    end

    println("done....")

end



