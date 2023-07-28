
import Base.Iterators

# Loads PPSeq module, and file IO wrappers.
include("./utils/all_utils.jl")

# Generate configs
random_seeds = 1:3
sequence_event_rates = (1.0,)
bkgd_rates = (15.0, 20.0, 25.0,)

itr = Iterators.product(
    random_seeds,
    sequence_event_rates,
    bkgd_rates
)

sweep_id = 1
@printf("[ time: %s ]\nStarting sweep %i\n", Dates.now(), sweep_id)
@printf("===============================\n", )

for (job_id, (seed, λ, bkgd_λ)) in enumerate(itr)

    @printf("===============================\n", )
    @printf("[ time: %s ]\n", Dates.now())
    @printf("Sweep %i, Running job %i / %i\n", sweep_id, job_id, length(itr))

    config = Dict(
        :sweep_id => sweep_id,
        :job_id => job_id,
        :dataset_id => "songbird",
        :random_seed => seed,
        :seq_type_conc_param => 1.0,
        :num_sequence_types =>  2,
        :mean_event_amplitude => 100.0,
        :var_event_amplitude => 1000.0,
        :neuron_response_conc_param => 0.1,
        :neuron_offset_pseudo_obs => 1.0,
        :neuron_offset_prior => 0.0,
        :neuron_width_pseudo_obs => 1.0,
        :num_warp_values => 1,
        :max_warp => 1.0,
        :warp_variance => 1.0,
        :neuron_width_prior => 0.3,
        :mean_bkgd_spike_rate => bkgd_λ,
        :var_bkgd_spike_rate => 3.0,
        :bkgd_spikes_conc_param => 1.0,
        :max_sequence_length => Inf,
        :seq_event_rate => λ,
        :num_anneals => 10,
        :samples_per_anneal => 100,
        :max_temperature => 40.0,
        :save_every_during_anneal => 999999999,
        :samples_after_anneal => 2000,
        :save_every_after_anneal => 10,
        :split_merge_moves_during_anneal => 10,
        :split_merge_moves_after_anneal => 1000,
        :split_merge_window => 1.0,
        :initialization => "background",
        :mask_lengths => 10.0,
        :percent_masked => 0.0,
    )

    load_train_save(config)

end
