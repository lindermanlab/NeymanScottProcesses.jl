## Structures

There are 4 structs we'll need to define to create a new Neyman-Scott model. The first two structs are immutable and correspond to the data and the priors of the model.

`Datapoint` -  a datapoint struct (e.g. `Spike`), which inherits `AbstractDatapoint`. This struct must contain all the information pertaining to a single datapoint, i.e. it's location and marks. For example, `Spike` contains `timestamp` and `neuron`.

`Priors` - a prior struct (e.g. `SeqPriors`), which inherits `AbstractPriors`. This struct can contain fixed model parameters or directly specify prior distributions.

The second two structs are mutable, and correspond to global variables and event-specific variables.

`Globals` - a global variable struct (e.g. `SeqGlobals`), which inherits `AbstractGlobals`. This struct contains the state of variables shared across events. This struct contains variables that are sampled during inference, but are not specific to a particular event. For example, the response width for each neuron during each sequence type is specified in `SeqGlobals.neuron_response_widths`.


`Event` - an event struct (e.g. `SeqEvent`), which inherits `AbstractEvent`. This struct contains both the sufficient statistics for an event containing datapoints {x_1, ..., x_m} and the sampled latent variables for the event. In `SeqEvent`, the sufficient statistics are `spike_count, summed_potentials, summed_precisions, summed_logZ, seq_type_posterior` and the sampled latent variables are `sampled_type, sampled_warp, sampled_timestamp, sampled_amplitude`.


## Methods

The major methods we need to define are those called in `gibbs_sample!`. In particular, these are

`log_posterior_predictive(event, datapoint, model)` - compute the posterior probability of assigning `datapoint` to `event`, conditioned on the current sufficient statistics for `event`, i.e. compute

$$log p ( x | T(x_1, ..., x_m), θ, z_k)$$

where `x` is a new spike, {x_1, ..., x_m} are a set of spikes that are currently assigned to a latent event `z_k`, T is the sufficient statistic, and `θ` are global parameters (neuron offsets, etc.).

`bkgd_log_like(datapoint, model)` - compute the posterior probability of assigning `datapoint` to the background

`log_posterior_predictive(datapoint, model)` - compute the posterior probability of assigning `datapoint` to a new event

`gibbs_sample_globals!(model, spikes, assignments)` - sample all the global variables in sequence, conditioned on the sampled latent events and datapoint assignments

`gibbs_sample_event!(event, model)` - sample the variables of a latent event, given its sufficient statistics and the model's global variables


There are some smaller methods that need implementation as well.

`reset!(event)` - reset the sufficient statistics and sampled values of the event, as if it were an empty event

`downdate_stats!(model, x, event)` - update `event`'s sufficient statistics after removing datapoint `x`

`update_stats!(model, x, event)` - update  `event`'s sufficient statistics after adding datapoint `x`

`set_posterior(model, event)` - cache information about the posterior distribution of event `event` based on the current sufficient statistics. This function is usually called after `update_stats!` and `downdate_stats!` and can greatly reduce computation time

`sample_globals(priors)` - sample an instance of `Globals` from `priors`

`sample_events(priors, globals)` - sample a list of events given `priors` and `globals`

`sample_data(model)` - sample a list of datapoints given `model`, which contains `priors`, `globals`, and `events`

`log_prior(globals, prior)` - compute the prior probability of the global parameters

`log_p_latents(model)` - compute the probability of the latent events,
given the global parameters and priors of the model

`log_like(model, spikes)` - compute the log likelihood of observed data, given the latent events, global variables, and model prior