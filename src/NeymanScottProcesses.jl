module NeymanScottProcesses

# ===
# CONSTANTS
# ===

const NOT_SAMPLED_AMPLITUDE = -2.0

# === 
# EXTENSIONS
# ===

import Base.Iterators
import Distributions

import StatsBase: sample



# ===
# IMPORTS
# ===

# Modules
using RecipesBase
using Statistics
using SparseArrays
# using StaticArrays
using IntervalSets: width, AbstractInterval

# Methods
using Distributions: cdf, mean, var
using LinearAlgebra: I, norm, logdet, det, eigvals
using Random: AbstractRNG, shuffle!
using SpecialFunctions: logabsgamma, logfactorial, gamma
using StatsBase: pweights, denserank, mean
using StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf

# Distributions
using Distributions: Categorical, Chisq, Normal, Poisson, TDist
using Distributions: Dirichlet, Multinomial, MultivariateNormal, InverseWishart
using Distributions: Product, Uniform, Distribution, ContinuousUnivariateDistribution
using PyPlot: PyObject, plt


# ===
# EXPORTS
# ===

# Distributions
export specify_gamma, mean, var
export RateGamma, NormalInvChisq, ScaledInvChiseq, SymmetricDirichlet

# Sampling and inference
export sample, log_prior, log_p_latents
export sample_full_process, plot!
export sample_random_spheres, split_data, sample_in_mask
export GibbsSampler, AnnealedSampler, MaskedSampler

# Models
export NeymanScottPriors, NeymanScottModel
export GaussianPriors, GaussianGlobals, GaussianCluster

# Regions
export Region, Box, Sphere


# ===
# INCLUDES
# ===

include("utils.jl")
include("distributions.jl")

# Core types
include("core/regions.jl")       # Define geometric regions.
include("core/model.jl")         # Main type definitions.
include("core/interface.jl")     # Rename?
include("core/cluster_list.jl")  # ClusterList struct functions.
include("core/likelihoods.jl")   # Likelihood function and other probability densities.
include("core/samples.jl")       # Methods to draw samples from the generative model.

# Samplers
include("samplers/base.jl")
include("samplers/gibbs.jl")
include("samplers/anneal.jl")
include("samplers/mask.jl")

# Models
include("models/gaussian.jl")

# Datasets
include("datasets/gaussian_testdata.jl")

# Plotting
# include("plots.jl")



# TODO
# - Reincorporate SparseMultinomial and SparseDirichletMultinomial to `distributions.jl`

# export log_joint
# export split_merge_sample!, annealed_masked_gibbs!
# export DistributedNeymanScottModel, make_distributed

# export Spike, SeqHypers, SeqGlobals, PPSeq
# export Cable, CablesPriors, CablesGlobals, CablesModel

end
