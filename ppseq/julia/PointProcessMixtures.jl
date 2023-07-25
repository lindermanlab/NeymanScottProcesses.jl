module PointProcessMixtures

# Exports
export NormInvWishartPrior, DirichletPrior, CrossPrior
export GaussianCluster, MultinomialCluster, NodeTimeMarkCluster
export DPMM, MFMM, NSPMM
export gibbs_sample, posterior, initialize_clusters

# Global namespace imports
using LinearAlgebra
using Statistics
using SparseArrays

import Base: size
import SpecialFunctions: logabsgamma
import StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf

# Local namespace imports
import Distributions
import Random
import Profile

const ds = Distributions
const rnd = Random

# Include files
# TODO put type definitions in a single file at the top, to avoid weird deps
include("./utils.jl")

include("./priors/base.jl")
include("./priors/niw.jl")
include("./priors/dirichlet.jl")
include("./priors/crossprod.jl")

include("./cluster_models/base.jl")
include("./cluster_models/gaussian.jl")
include("./cluster_models/multinomial.jl")
include("./cluster_models/nodetimemark.jl")
include("./cluster_models/initialize.jl")

include("./mixture_models/base.jl")
include("./mixture_models/dpmm.jl")
include("./mixture_models/mfmm.jl")
include("./mixture_models/nspmm.jl")

include("./gibbs.jl")

end
