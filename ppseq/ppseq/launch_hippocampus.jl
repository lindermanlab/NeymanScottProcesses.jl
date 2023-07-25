
import Base.Iterators

# Loads PPSeq module, and file IO wrappers.
include("./utils/all_utils.jl")

# HERE IS WHAT LOOKS BY EYE TO BE A GOOD SET OF HYPERPARAMETERS.

base_config = Dict{Symbol,Union{Int64,Float64,String}}(
    :seq_type_conc_param => 3.0,
    :num_sequence_types =>  2,
    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,
    :neuron_response_conc_param => 1.0,
    :neuron_offset_pseudo_obs => 0.15,
    :neuron_offset_prior => 0.0,
    :neuron_width_pseudo_obs => 4.0,
    :neuron_width_prior => 0.1,
    :num_warp_values => 5,
    :warp_variance => 1.0,
    :mean_bkgd_spike_rate => 500.0,
    :var_bkgd_spike_rate => 200.0,
    :bkgd_spikes_conc_param => 1.0,
    :max_sequence_length => 30.0,
    :seq_event_rate => .001,
)

# spikes, max_time, num_neurons = load_dataset("hippocampus")
# model = construct_model(config, max_time, num_neurons)
# BSON.@load joinpath(DATAPATH, "hippocampus/maze_positions.bson") pos_x pos_times

# # Draw sample.
# (
#     sampled_spikes,
#     sampled_assignments,
#     sampled_events
# ) =
# sample(
#     model;
#     resample_latents=true,
#     resample_globals=false
# )

# plot_raster(
#     sampled_spikes,
#     sampled_events,
#     sampled_assignments,
#     sortperm_neurons(model.globals);
#     lw=0
# )
# plt.xlim(1000, 1300)
# plt.title("sampled from prior")

# plot_raster(spikes; lw=0, color="k")
# plt.xlim(1000, 1300)
# plt.title("real data")

# fig, ax = plt.subplots(2, 1)
# ax[1].hist(collect(values(countmap([s.neuron for s in sampled_spikes]))), collect(0:100:10000))
# ax[1].set_title("spikes per neuron, sampled from prior")
# ax[2].hist(collect(values(countmap([s.neuron for s in spikes]))), collect(0:100:10000))
# ax[2].set_title("spikes per neuron, real data")

# Generate configs
random_seeds = 1:5
max_warps = (1.0, 1.5)
init_strategies = ("convnmf", "background")

itr = Iterators.product(
    max_warps,
    init_strategies,
    random_seeds
)

sweep_id = 5
@printf("[ time: %s ]\nStarting sweep %i\n", Dates.now(), sweep_id)
@printf("===============================\n", )

config = deepcopy(base_config)

for (job_id, (mx_wrp, init, seed)) in enumerate(itr)

    @printf("===============================\n", )
    @printf("[ time: %s ]\n", Dates.now())
    @printf("Sweep %i, Running job %i / %i\n", sweep_id, job_id, length(itr))

    config = deepcopy(base_config)

    config[:sweep_id] = sweep_id
    config[:job_id] = job_id
    config[:dataset_id] = "hippocampus"
    config[:random_seed] = seed
    config[:num_threads] = 0

    config[:max_warp] = mx_wrp
    config[:num_anneals] = (init == "convnmf") ? 0 : 15
    config[:samples_per_anneal] = 100
    config[:max_temperature] = 100.0

    config[:samples_per_resample] = 1
    config[:num_spike_resamples_per_anneal] = 100
    config[:num_spike_resamples_after_anneal] = 300

    config[:save_every_during_anneal] = 10
    config[:samples_after_anneal] = 50
    config[:save_every_after_anneal] = 1
    config[:split_merge_moves_during_anneal] = 100
    config[:split_merge_moves_after_anneal] = 1000
    config[:split_merge_window] = 5.0

    config[:initialization] = init
    config[:convnmf_seqlen] = 10.0
    config[:convnmf_binsize] = 1.0
    config[:convnmf_peak_thres] = 0.5
    config[:convnmf_bkgd_thres] = 0.2

    config[:mask_data] = "randomly"
    config[:mask_lengths] = 30.0
    config[:percent_masked] = 5.0

    results = load_train_save(config)

end
