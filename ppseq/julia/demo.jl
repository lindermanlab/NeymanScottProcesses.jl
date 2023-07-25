# == Bunch of imports == #
using LinearAlgebra
using Statistics

import Distributions
import PyPlot
import Random

const ds = Distributions
const plt = PyPlot
const rnd = Random

# This is a hack to import the module for now, we can
# make PointProcessMixtures a full package later.
!(pwd() in LOAD_PATH) && push!(LOAD_PATH, pwd())
using PointProcessMixtures


function gaussian_demo(
        n_clusters, n_datapts, n_gibbs_samples;
        method="dpmm", seed=1234
    )

    # Set random seed.
    rnd.seed!(seed)

    # Make some data.
    data = Vector{Float64}[]

    cluster_centers = 10 * randn(2, n_clusters)
    for i = 1:n_datapts
        push!(data, randn(2) + cluster_centers[:, (i % n_clusters) + 1])
    end

    # Prior distribution on cluster parameters.
    m = zeros(2)
    m_n = 0.01
    S = [ 10.0     0.0
            0.0   10.0 ]
    S_n = 3.01
    prior = NormInvWishartPrior(vec(m), m_n, S, S_n)

    # Initial cluster assignments.
    num_init_clusters = 1
    _a, _c = initialize_clusters(
        data, num_init_clusters, prior, GaussianCluster)

    # Specify model.
    if method == "dpmm"

        alpha = 10.0
        model = DPMM(alpha, prior, _a, _c)

    elseif method == "mfmm"

        gamma = 1.0
        k_distrib = ds.Poisson(20.0)
        model = MFMM(gamma, k_distrib, prior, _a, _c)

    elseif method == "nspmm"

        min_datapoints = n_datapts - 1
        max_datapoints = n_datapts + 1
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

    # Run Gibbs sample.
    @time z, model, log_like_hist = gibbs_sample(model, data, n_gibbs_samples)

    # Plot
    plt.figure()
    plt.plot(log_like_hist)
    plt.ylabel("log likelihood")
    plt.xlabel("iteration")

    x = [d[1] for d in data]
    y = [d[2] for d in data]

    plt.figure()
    plt.scatter(x, y, c=z[:, 1], cmap="Set1_r")
    plt.title("First sample")

    plt.figure()
    plt.scatter(x, y, c=z[:, end], cmap="Set1_r")
    plt.title("Last sample")

end


function multinomial_demo(
        n_clusters, n_obs, n_features, n_gibbs_samples;
        n_per_obs=500, method="dpmm", seed=1234
    )

    # Set random seed.
    rnd.seed!(seed)

    # Make some data.
    cluster_centers = rand(
        ds.Dirichlet(ones(n_features)), n_clusters)

    multinomials = [
        ds.Multinomial(n_per_obs, cluster_centers[:, j]) for j in 1:n_clusters]

    data = Vector{Int}[]

    for i = 1:n_obs
        push!(data, ds.rand(multinomials[(i % n_clusters) + 1]))
    end

    # Prior distribution on cluster parameters.
    prior = DirichletPrior(ones(n_features))

    # Initial cluster assignments.
    num_init_clusters = 1
    _a, _c = initialize_clusters(
        data, num_init_clusters, prior, MultinomialCluster)

    # Specify model.
    if method == "dpmm"

        alpha = 10.0
        model = DPMM(alpha, prior, _a, _c)

    elseif method == "mfmm"

        gamma = 1.0
        k_distrib = ds.Poisson(20.0)
        model = MFMM(gamma, k_distrib, prior, _a, _c)

    end

    # Run Gibbs sample.
    @time z, model, log_like_hist, num_cluster_hist = gibbs_sample(model, data, n_gibbs_samples)

    # Plot log-likelihood over time.
    plt.figure()
    plt.plot(log_like_hist)
    plt.ylabel("log likelihood")
    plt.xlabel("iteration")

    plt.figure()
    plt.plot(num_cluster_hist)
    plt.ylabel("number of clusters")
    plt.xlabel("iteration")

    post = ds.Dirichlet(posterior(model.clusters[1]).alpha)
    for i = 1:n_clusters
        plt.figure()
        plt.plot(mean(post))
        plt.plot(cluster_centers[:, i])
    end

    return model

end
