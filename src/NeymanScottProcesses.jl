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
using Distributions: cdf, mean, var, logpdf, rand!
using LinearAlgebra: norm, logdet, det
using Random: AbstractRNG, shuffle!
using SpecialFunctions: logabsgamma, logfactorial, gamma
using StatsBase: pweights, denserank, mean
using StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf

# Distributions
using Distributions: Categorical, Chisq, Normal, Poisson, TDist
using Distributions: Dirichlet, MultivariateNormal, InverseWishart
using Distributions: DirichletMultinomial, Multinomial
using Distributions: NegativeBinomial




# ===
# EXPORTS
# ===

# Distributions
export specify_gamma, specify_inverse_gamma, mean, var
export RateGamma, InverseGamma, NormalInvChisq, ScaledInvChiseq, SymmetricDirichlet

# Sampling and inference
export sample, log_prior, log_p_latents
export create_random_mask, split_data_by_mask, sample_masked_data
export GibbsSampler, Annealer, MaskedSampler, ReversibleJumpSampler

# Helper functions
export cooccupancy_matrix

# Models
export GaussianNeymanScottModel, GaussianPriors, GaussianGlobals, GaussianCluster, RealObservation
export CablesModel, CablesPriors, CablesGlobals, CableCluster, Cable

# Masks
export CircleMask, CablesMask



# ===
# INCLUDES
# ===

include("utils.jl")
include("distributions.jl")

# Core types
include("core/abstract.jl")  # Abstract types and basic functionality
include("core/nsp.jl")  # Neyman-Scott Model
include("core/interface.jl")  # Interface that models must implement
include("core/cluster_list.jl")
include("core/mask.jl")

# Models
include("models/gaussian.jl")
include("models/cables.jl")

# Samplers
include("samplers/base.jl")
include("samplers/gibbs.jl")
include("samplers/anneal.jl")
include("samplers/mask.jl")
include("samplers/rjmcmc.jl")
include("samplers/split_merge.jl")

# Plotting and diagnostics
include("diagnostic.jl")
include("plots.jl")



# TODO
# - split_merge_sample!
# - DistributedNeymanScottModel, make_distributed
# - Spike, SeqHypers, SeqGlobals, PPSeq

end
