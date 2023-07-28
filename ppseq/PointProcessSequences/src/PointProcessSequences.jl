module PointProcessSequences

# === IMPORTS === #
using LinearAlgebra
using Statistics
using SparseArrays

import Base: size, rand, length, getindex, iterate, in
import Base.Iterators
import StatsBase: sample, pweights, denserank, mean
import Random
import SpecialFunctions
import SpecialFunctions: logabsgamma, logfactorial
import StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf
import Distributions
import Distributions: cdf

# === EXPORTS === #

export log_joint, log_prior, log_p_latents
export SymmetricDirichlet, RateGamma, ScaledInvChisq, NormalInvChisq
export specify_gamma, sample, mean, var

# Gibbs sampling
export split_merge_sample!, gibbs_sample!, annealed_gibbs!
export masked_gibbs!, annealed_masked_gibbs!, Mask
export create_random_mask, split_data_by_mask, create_blocked_mask, sample_masked_data!, sample_masked_data

# Distributed model
export DistributedNeymanScottModel, make_distributed

# Diagnostic tools
export background_assignment_prob, new_cluster_assignment_prob, prob_ratio_vs_bkgd_temp, prob_ratio_vs_event_temp

# Models
export Spike, EventSummaryInfo, SeqHypers, SeqGlobals, PPSeq, DistributedPPSeq
export Point, GaussianEventSummary, GaussianPriors, GaussianGlobals, GaussianNeymanScottModel 
export Cable, CablesEventSummary, CablesPriors, CablesGlobals, CablesModel

# === UTILS === #

include("./utils/misc.jl")
include("./utils/distributions.jl")

# === MODEL === #

include("./model/base.jl")
include("./model/eventlist.jl")
include("./model/nsp.jl")
include("./model/distributed.jl")
include("./processes.jl")

 # Model structs
 # - PPSeq : holds full model
 # - SeqEvent : holds suff stats for a latent event
 # - EventList : dynamically re-sized array of latent events.
 # - SeqPriors : holds prior distributions.
 # - SeqGlobals : holds global variables.
include("./model/ppseq/structs.jl")

 # Convience methods for constructing / accessing PPSeq model struct.
include("./model/ppseq/model.jl")

 # Methods for creating latent events, adding and removing spikes
 # and updating sufficient statistics.
include("./model/ppseq/events.jl")

 # Evaluate various probability distributions for the model:
 # - predictive posterior
 # - prior distribution on global parameters
 # - distribution on latent events, given global parameters
 # - joint distribution on global parameters, latent events, observed spikes
include("./model/ppseq/probabilities.jl")



# Gibbs event and global updates
include("./model/ppseq/gibbs.jl")

include("./model/gaussian/structs.jl")
include("./model/gaussian/events.jl")
include("./model/gaussian/probabilities.jl")
include("./model/gaussian/model.jl")
include("./model/gaussian/gibbs.jl")

include("./model/cables/structs.jl")
include("./model/cables/events.jl")
include("./model/cables/probabilities.jl")
include("./model/cables/model.jl")
include("./model/cables/gibbs.jl")


# === PARAMETER INFERENCE === #

 # Collapsed Gibbs sampling.
include("./algorithms/gibbs.jl")

 # Evaluate log likelihood on heldout data.
include("./model/masked_probabilities.jl")

 # Masked Gibbs sampling (for cross-validation).
include("./algorithms/masked_gibbs.jl")

 # Distributed collapsed Gibbs sampling
include("./algorithms/distributed_gibbs.jl")

 # Collapsed Gibbs sampling with annealing.
include("./algorithms/annealed_gibbs.jl")

 # Split merge sampler.
include("./algorithms/split_merge.jl")

# Diagnostic tools
include("./utils/diagnostics.jl")

# === END OF MODULE === #

end
