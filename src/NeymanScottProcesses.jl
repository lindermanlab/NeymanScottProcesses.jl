module NeymanScottProcesses


# === 
# EXTENSIONS
# ===

import Base.Iterators
import Distributions


# ===
# IMPORTS
# ===

# Modules
using LinearAlgebra
using Statistics
using SparseArrays

# Methods
using Distributions: cdf, mean, var
using Random: AbstractRNG
using SpecialFunctions: logabsgamma, logfactorial
using StatsBase: sample, pweights, denserank, mean
using StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf

# Distributions
using Distributions: Categorical, Chisq, Normal, Poisson, TDist
using Distributions: Dirichlet, Multinomial, MultivariateNormal, InverseWishart

export RateGamma, NormalInvChisq, ScaledInvChiseq, SymmetricDirichlet
export specify_gamma, mean, var


# ===
# INCLUDES
# ===

include("utils.jl")
include("distributions.jl")





# TODO
# - [ ] Reincorporate SparseMultinomial and SparseDirichletMultinomial to `distributions.jl`

# import Base: size, rand, length, getindex, iterate, in

# export log_joint, log_prior, log_p_latents
# export split_merge_sample!, gibbs_sample!, annealed_gibbs!
# export masked_gibbs!, annealed_masked_gibbs!, Mask
# export create_random_mask, split_data_by_mask, create_blocked_mask, sample_masked_data!, sample_masked_data
# export DistributedNeymanScottModel, make_distributed
# export background_assignment_prob, new_cluster_assignment_prob, prob_ratio_vs_bkgd_temp, prob_ratio_vs_event_temp

# export Spike, EventSummaryInfo, SeqHypers, SeqGlobals, PPSeq, DistributedPPSeq
# export Point, GaussianEventSummary, GaussianPriors, GaussianGlobals, GaussianNeymanScottModel 
# export Cable, CablesEventSummary, CablesPriors, CablesGlobals, CablesModel

end
