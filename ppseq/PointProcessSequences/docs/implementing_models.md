# Creating a New Model

```@meta
CurrentModule = PointProcessSequences
```

## Directory Structure

To start developing a new model, create a new folder in `src/model/` with the name of your model. In the documentation below, we will refer to the model as `MyModel` and use the folder `src/model/mymodel`.


## Types

To create a new Neyman-Scott model, we need to 4 composite types (i.e. structs) that define the basic behavior of the model in `mymodel/structs.jl`. These are

```julia
const DIM = 2

# (Observed) datapoint
struct MyDatapoint <: AbstractDatapoint{DIM}
    # -- Mandatory fields --
    position::Vector{Float64}  # Must satisfy length(position) == DIM

    # -- Place custom fields below --
end

# (Latent) events (cluster, sequence, latent variable, etc)
struct MyEvent <: AbstractEvent{MyDatapoint}
    # -- Mandatory fields --
    datapoint_count::Int
    sampled_position::Vector{Float64}  # Must satisfy length(position) == DIM
    sampled_amplitude::Float64

    # -- Place custom fields below --
end
    
# Global variables
struct MyGlobals <: AbstractGlobals
    # -- Mandatory fields --
    bkgd_rate::Float64
    
   # -- Place custom fields below --
end 
   

struct MyPriors <: AbstractPriors
    # -- Mandatory fields --
    event_rate::Float64
    event_amplitude::RateGamma
    bkgd_amplitude::RateGamma   
    
    # -- Place custom fields below --
end
```

Each composite type refers to a particular element of the Neyman-Scott model. In particular, `MyDatapoint` refers to observed datapoints that are fed in as inputs to the model. `MyEvent` holds information about latent events (i.e. clusters) that are to be inferred during sampling. Finally, `MyGlobals` holds global variables (i.e. latent variables that are not specific to a particular event or datapoint) and `MyPriors` holds all the priors of the model.

In addition to their mandatory fields, the types may also contain custom fields specific to the model of interest. For example, in a neuroscience dataset, one might wish to add the field `neuron::Int` to `MyDatapoint`.

To connect all these types together, we define the Neyman-Scott model incorporating them:

```julia
# It's convenient to use a type-alias for easy reference
const MyModel = NeymanScottModel{DIM, MyDatapoint, MyEvent, MyGlobals, MyPriors}
```

Next, we will need to define the basic behavior of these types. This will require specifying a number of simple functions that are used in generative sampling and posterior inference via collapsed Gibbs sampling.


## Event Mechanics

We'll begin by specifying (latent) event behavior. Our struct `MyEvent` contains two types of information: sufficient statistics used during the collapsed sampling of the parent assignments and sampled values used during the (not collapsed) sampling of the global variables. We'll place all this behavior in `mymodel/events.jl`. We'll need to specify all the methods documented below.


```@docs
AbstractEvent(args...)
constructor_args(event)
reset!(event)
been_sampled(event)
too_far(x, event, model)

set_posterior!(model, k)
remove_datapoint!(model, point, event_index)
add_datapoint!(model, point, event_index)

event_list_summary(model::AbstractModel)
```

## Probabilities

The next step is to specify probabilities associated with the model in `mymodel/probabilities.jl`. These are used both during posterior inference and when evaluating the likelihood of the observed data under the model.

```@docs
bkgd_log_like(m::AbstractModel, x::AbstractDatapoint)
log_posterior_predictive(e::AbstractEvent, x::AbstractDatapoint, m::AbstractModel)
log_posterior_predictive(x::AbstractDatapoint, m::AbstractModel) 
bkgd_intensity(m::AbstractModel, x::AbstractDatapoint)
event_intensity(m::AbstractModel, e::AbstractEvent, x::AbstractDatapoint)
log_p_latents(m::AbstractModel)
log_prior(model::AbstractModel)
```


## Defining and Sampling Models

In `mymodel/model.jl`, we specify the model constructor as well as sampling methods for the model.

```@docs
AbstractModel()
sample(model::AbstractModel; resample_latents::Bool=false, resample_globals::Bool=false)
sample(priors::AbstractPriors)
```

## Inference via Gibbs Sampling

In `mymodel/gibbs.jl`, we define the Gibbs update rules for updating global variables and latent events.

```@docs
gibbs_sample_globals!(m::AbstractModel, data::Vector{<: AbstractDatapoint}, assignments::Vector{Int})
gibbs_sample_event!(e::AbstractEvent, m::AbstractModel)
```