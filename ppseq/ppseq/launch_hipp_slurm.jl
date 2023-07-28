
import Base.Iterators

# Loads PPSeq module, and file IO wrappers.
include("./utils/all_utils.jl")

# HERE IS WHAT LOOKS BY EYE TO BE A GOOD SET OF HYPERPARAMETERS.
config = Dict{Symbol,Union{Int64,Float64,String}}(
    :dataset_id => "hippocampus",
    :seq_type_conc_param => 3.0,
    :num_sequence_types =>  2,
    :neuron_response_conc_param => 1.0,
    :neuron_offset_prior => 0.0,
    :neuron_width_pseudo_obs => 4.0,
    :num_warp_values => 20,
    :warp_variance => 1.0,
    :bkgd_spikes_conc_param => 1.0,
    :max_sequence_length => 50.0,
)

# Job and sweep id numbers.
job_id = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])
sweep_id = 503

# Generate config
Random.seed!(job_id)
mx_wrp = exp10(rand(Uniform(0.0, log10(2.0))))
seq_amp = exp10(rand(Uniform(1.5, 3)))
bkgd = exp10(rand(Uniform(2, 4)))
seq_rate = exp10(rand(Uniform(-3, -1)))
init = sample(["convnmf", "background"])
wdth = exp10(rand(Uniform(-1, 1)))
offobs = exp10(rand(Uniform(-2, 1)))
seed = sample(collect(1:5))  # limit number of random masks

config[:sweep_id] = sweep_id
config[:job_id] = job_id
config[:random_seed] = seed
config[:max_warp] = mx_wrp
config[:mean_event_amplitude] = seq_amp
config[:var_event_amplitude] = seq_amp
config[:mean_bkgd_spike_rate] = bkgd
config[:var_bkgd_spike_rate] = bkgd
config[:seq_event_rate] = seq_rate
config[:initialization] = init
config[:neuron_width_prior] = wdth
config[:neuron_offset_pseudo_obs] = offobs

config[:num_threads] = 0
config[:num_anneals] = (init == "convnmf") ? 0 : 30
config[:samples_per_anneal] = 100
config[:max_temperature] = 500.0

config[:samples_per_resample] = 1
config[:num_spike_resamples_per_anneal] = 100
config[:num_spike_resamples_after_anneal] = 300

config[:save_every_during_anneal] = 10
config[:samples_after_anneal] = 50
config[:save_every_after_anneal] = 1
config[:split_merge_moves_during_anneal] = 100
config[:split_merge_moves_after_anneal] = 1000
config[:split_merge_window] = 5.0

config[:convnmf_seqlen] = 10.0
config[:convnmf_binsize] = 1.0
config[:convnmf_peak_thres] = 0.5
config[:convnmf_bkgd_thres] = 0.2

config[:mask_data] = "randomly"
config[:mask_lengths] = 30.0
config[:percent_masked] = 7.5

# Log info.

@printf("===============================\n", )
@printf("[ start time: %s ]\nStarting sweep %i\n", Dates.now(), sweep_id)
@printf("Running job %i\n", job_id)

for (k, v) in config
    println(k, " => ", v)
end

# Run training.
results = load_train_save(config)

@printf("===============================\n", )
@printf("[ end time: %s ]\n", Dates.now())
