module NeymanScottProcesses


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
using StaticArrays

# Methods
using Distributions: cdf, mean, var
using LinearAlgebra: norm, logdet, det
using Random: AbstractRNG, shuffle!
using SpecialFunctions: logabsgamma, logfactorial, gamma
using StatsBase: pweights, denserank, mean
using StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf

# Distributions
using Distributions: Categorical, Chisq, Normal, Poisson, TDist
using Distributions: Dirichlet, Multinomial, MultivariateNormal, InverseWishart




# ===
# EXPORTS
# ===

# Distributions
export specify_gamma, mean, var
export RateGamma, NormalInvChisq, ScaledInvChiseq, SymmetricDirichlet

# Sampling and inference
export sample, log_prior, log_p_latents
export create_random_mask, split_data_by_mask, sample_masked_data
export GibbsSampler, Annealer, MaskedSampler

# Models
export GaussianNeymanScottModel, GaussianPriors, GaussianGlobals, GaussianCluster, RealObservation




# ===
# INCLUDES
# ===

include("utils.jl")
include("distributions.jl")

# Core types
include("core/abstract.jl")  # Abstract types and basic functionality
include("core/nsp.jl")  # Neyman-Scott Model
include("core/interface.jl")  # Interface that models must implement
<<<<<<< HEAD
include("core/eventlist.jl")  # Managing (non-parametric) events
include("core/mask.jl")
=======
include("core/cluster_list.jl")  # Managing (non-parametric) events
>>>>>>> rename latent "events" to "clusters"

# Samplers
include("samplers/base.jl")
include("samplers/gibbs.jl")
include("samplers/anneal.jl")
include("samplers/mask.jl")

# Models
include("models/gaussian.jl")

# Plotting
include("plots.jl")



# TODO
# - Reincorporate SparseMultinomial and SparseDirichletMultinomial to `distributions.jl`

# export log_joint
# export split_merge_sample!, annealed_masked_gibbs!
# export DistributedNeymanScottModel, make_distributed

# export Spike, EventSummaryInfo, SeqHypers, SeqGlobals, PPSeq
# export Cable, CablesEventSummary, CablesPriors, CablesGlobals, CablesModel

end
