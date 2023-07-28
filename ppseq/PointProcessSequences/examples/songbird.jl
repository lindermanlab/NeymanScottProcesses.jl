using Revise

using LinearAlgebra
using Statistics
using PointProcessSequences
using Printf

import MAT: matopen
import PyPlot: plt
import BSON

import Distributions
import Random
const ds = Distributions
const rnd = Random


"""
Loads deconvolved calcium spikes from zebra finch HVC.
Data collected by Emily Mackevicius in Michale Fee's
lab (MIT) and available at: https://github.com/FeeLab/seqNMF
Returns
-------
spikes :
    Vector of Spike structs holding (neuron_id, timestamp)
    for each spike.
max_time : 
    Real number > 0 specifying length of recording.
num_neurons :
    Integer specifying number of simultaneously recorded cells.
"""
function load_songbird_data()
    file = matopen("./examples/MackeviciusData.mat")

    neural = read(file, "NEURAL")
    fs = read(file, "VIDEOfs")

    num_neurons, T_bins = size(neural)
    t_bins = (1:T_bins) / fs
    max_time = T_bins / fs

    n_spks = []
    j_spks = []
    for n in 1:num_neurons
        for j in 1:T_bins
            if (neural[n, j] > 0)
                push!(n_spks, n)
                push!(j_spks, j)
            end
        end
    end
    t_spks = t_bins[j_spks]

    # Known permutation to generate sequences
    perm = [0, 12, 21, 24, 28, 29, 39, 46, 70, 72, 74, 14, 65,  3, 36, 57, 45,
        10,  2, 26, 40, 54, 50, 62,  9, 37, 63, 35, 66,  5, 32, 38, 41, 68,
        69, 61, 16, 11, 56, 33, 55, 60,  4, 18, 19, 31, 27, 30, 42, 23, 47,
        48, 67, 17, 43, 44, 52, 53, 71, 13, 22, 51,  7,  8, 59,  6, 15,  1,
        73, 64, 49, 20, 25, 34, 58]

    n_spks = sortperm(perm)[n_spks]
    spikes = [Spike(n, t) for (n, t) in zip(n_spks, t_spks)]

    # Sort spikes by timestamp
    spikes = [spikes[i] for i in sortperm([x.timestamp for x in spikes])]


    close(file)
    return spikes, max_time, num_neurons
end

"""
Loads PP-Seq model with hyperparameters tuned for songbird HVC.
"""
function load_songbird_model(
    max_time::Float64,
    num_neurons::Int64
)

    # Constants.
    num_sequence_types = 2
    max_sequence_events = 1000
    max_sequence_length = Inf  # seconds

    # Expected number of sequence events per second.
    seq_event_rate = 1.0

    # Prior on sequence type proportions / relative frequencies.
    seq_type_proportions = SymmetricDirichlet(1.0, num_sequence_types)

    # Prior on expected number of spikes induces by a sequence events.
    seq_event_amplitude = specify_gamma(
        30.0,         # mean of gamma; α / β
        5.0^2        # variance of gamma; α / β²
    )

    # Prior on relative response amplitudes per neuron to each sequence type.
    neuron_response_proportions = SymmetricDirichlet(0.1, num_neurons)

    # Prior on expected number of background spikes in the entire recording.
    bkgd_amplitude = specify_gamma(   
        15.0,                 # mean of gamma; α / β
        1e-5                 # variance of gamma; α / β²
    )

    # Prior on relative background firing rates across neurons.
    bkgd_proportions = SymmetricDirichlet(1.0, num_neurons)

    # Prior on the response offsets and widths for each neuron.
    neuron_response_profile = NormalInvChisq(
        1.0,   # κ, pseudo-observations of prior mean
        0.0,   # m, prior mean for offset parameter
        2.0,   # ν, pseudo-observations of prior variance
        0.3    # s2, prior variance
    )

    return PPSeq(
        # constants
        max_time,
        max_sequence_length,

        # priors
        seq_event_rate,  # this gets converted to K ~ Poisson(max_time * rate)
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )

end


function fit_songbird_annealed(
        ;
        plots::Bool=true,
    )

    # Load data.
    spikes, max_time, num_neurons = load_songbird_data()

    # Load model.
    model = load_songbird_model(
        max_time,
        num_neurons
    )

    # Draw annealed Gibbs samples.
    (
        assignments,
        assignment_hist,
        annealed_log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    annealed_gibbs!(
        model,
        spikes,
        fill(-1, length(spikes)),
        3,    # num anneals
        200,   # samples per anneal
        5.0^2,  # max temperature
        0,     # num split merge moves
        1.0,   # split merge window
        10,     # save every
        verbose=true
    )

    # Draw regular Gibbs samples
    (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    gibbs_sample!(
        model,
        spikes,
        assignment_hist[:, end],
        500,
        0,
        1.0,
        10,
        verbose=true
    )


    colors = ["r", "b"]
    spike_colors = [
        p == -1 ? "k" : colors[model.sequence_events[p].sampled_type]
        for p in assignment_hist[:, end]
    ]

    if plots
        plt.figure()
        plt.plot([annealed_log_p_hist[2:end]; log_p_hist])
        plt.ylabel("log likelihood")

        plt.figure()
        plt.plot([length(unique(assignment_hist[:, i])) - 1 for i=1:size(assignment_hist, 2)])
        plt.ylabel("num sequence events")

        plt.figure()
        plt.scatter(
            [s.timestamp for s in spikes], 
            [s.neuron for s in spikes], 
            c=spike_colors  ,
            s=4
        )
    end

    return model, assignment_hist, log_p_hist
end