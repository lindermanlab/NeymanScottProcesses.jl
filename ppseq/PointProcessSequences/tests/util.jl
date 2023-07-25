using PointProcessSequences

function make_test_spikes_and_model(;
    num_spikes::Int64=100,
    max_time::Float64=100.0,
    num_neurons::Int64=10,
    num_sequence_types::Int64=2,
    num_warp_values::Int64=1,
    max_warp::Float64=1.0,
    warp_variance::Float64=1.0,
)

# Make spikes.
ns = rand(1:num_neurons, num_spikes)
ts = rand(num_spikes) * max_time
spikes = [Spike(n, t) for (n, t) in zip(ns, ts)]

# Prior on sequence type proportions / relative frequencies.
seq_type_proportions = SymmetricDirichlet(1.0, num_sequence_types)

# Prior on expected number of spikes induces by a sequence events.
seq_event_amplitude = specify_gamma(100.0, 100.0)

# Prior on relative response amplitudes per neuron to each sequence type.
neuron_response_proportions = SymmetricDirichlet(1.0, num_neurons)

# Prior on the response offsets and widths for each neuron.
neuron_response_profile = NormalInvChisq(2.0, 0.0, 3.0, 1.0)

# Prior on expected number of background spikes in a unit time interval.
bkgd_amplitude = specify_gamma(5.0, 5.0)

# Prior on relative background firing rates across neurons.
bkgd_proportions = SymmetricDirichlet(1.0, num_neurons)

seq_event_rate = 1.0
max_sequence_length = Inf

model = PPSeq(
    # constants
    max_time,
    max_sequence_length,

    # warp parameters
    num_warp_values,
    max_warp,
    warp_variance,

    # priors
    seq_event_rate,
    seq_type_proportions,
    seq_event_amplitude,
    neuron_response_proportions,
    neuron_response_profile,
    bkgd_amplitude,
    bkgd_proportions
)

return model, spikes

end