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
using SpecialFunctions: logabsgamma, logfactorial
using StatsBase: pweights, denserank, mean
using StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf

# Distributions
using Distributions: Categorical, Chisq, Normal, Poisson, TDist
using Distributions: Dirichlet, Multinomial, MultivariateNormal, InverseWishart




# ===
# EXPORTS
# ===

export RateGamma, NormalInvChisq, ScaledInvChiseq, SymmetricDirichlet
export specify_gamma, mean, var

export sample, log_prior, log_p_latents
export GibbsSampler

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
include("core/eventlist.jl")  # Managing (non-parametric) events

# Samplers
include("samplers/base.jl")
include("samplers/gibbs.jl")

# Models
include("models/gaussian.jl")

# Plotting
include("plots.jl")



# TODO
# - [ ] Reincorporate SparseMultinomial and SparseDirichletMultinomial to `distributions.jl`
# export log_joint
# export split_merge_sample!, gibbs_sample!, annealed_gibbs!
# export masked_gibbs!, annealed_masked_gibbs!, Mask
# export create_random_mask, split_data_by_mask, create_blocked_mask, sample_masked_data!, sample_masked_data
# export DistributedNeymanScottModel, make_distributed

# export Spike, EventSummaryInfo, SeqHypers, SeqGlobals, PPSeq, DistributedPPSeq
# export Cable, CablesEventSummary, CablesPriors, CablesGlobals, CablesModel

end
