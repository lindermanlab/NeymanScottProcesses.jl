"""
Distributed Neyman Scott Model
"""
mutable struct DistributedNeymanScottModel{M} <: AbstractModel where M <: NeymanScottModel
    primary_model::M
    num_partitions::Int64
    submodels::Vector{M}
end


# ===========================
# ==== Inherit Primary Model Behavior
# ===========================


function Base.getproperty(model::DistributedNeymanScottModel, sym::Symbol)
    # We must explicitly call getfield (the default behavior of getproperty)
    # to avoid the recursion.
    primary_model = getfield(model, :primary_model)

    if hasfield(typeof(primary_model), sym)
        return Base.getproperty(primary_model, sym)
    else
        return getfield(model, sym)
    end
end


# ===========================
# ==== Distributed Model Constructors
# ===========================


"""Distributed `primary_model` across `num_partitions` partitions."""
function make_distributed(primary_model::NeymanScottModel, num_partitions::Int64)
    # Create submodels
    submodels = NeymanScottModel[]
    for part = 1:num_partitions
        push!(submodels, deepcopy(primary_model))
    end

    # Pass globals and priors to the submodels
    # This way, the global variables and priors refer to the same objects
    for submodel in submodels
        submodel.priors = primary_model.priors
        submodel.globals = primary_model.globals
    end

    return DistributedNeymanScottModel(primary_model, num_partitions, submodels)
end