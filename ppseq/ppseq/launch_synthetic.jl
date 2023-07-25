import Base.Iterators

# Loads PPSeq module, and file IO wrappers.
include("./utils/all_utils.jl")

# Load ground truth config.
config = Dict{Symbol,Union{Int64,Float64,String}}(
    Symbol(key) => val for (key, val) in YAML.load(open(joinpath(DATAPATH, "synthetic", "config.yml")))
)

# Generate configs
random_seeds = 1:5
num_sequence_types = (1, 2, 3, 4)
mask_strategies = ("randomly", "in blocks")

itr = Iterators.product(
    random_seeds,
    num_sequence_types,
    mask_strategies
)

sweep_id = 3
@printf("[ time: %s ]\nStarting sweep %i\n", Dates.now(), sweep_id)
@printf("===============================\n", )

for (job_id, (seed, R, msk)) in enumerate(itr)
    
    @printf("===============================\n", )
    @printf("[ time: %s ]\n", Dates.now())
    @printf("Sweep %i, Running job %i / %i\n", sweep_id, job_id, length(itr))

    # Update config.
    config[:sweep_id] = sweep_id
    config[:job_id] = job_id
    config[:seed] = seed
    config[:num_sequence_types] = R

    # Add details to config about inference.
    config[:num_anneals] = 10
    config[:samples_per_anneal] = 100
    config[:max_temperature] = 1e4
    config[:save_every_during_anneal] = 10
    config[:samples_after_anneal] = 250
    config[:save_every_after_anneal] = 10
    config[:split_merge_moves_during_anneal] = 10
    config[:split_merge_moves_after_anneal] = 10
    config[:split_merge_window] = 3.0
    config[:initialization] = "background"
    config[:convnmf_seqlen] = 3.0
    config[:convnmf_peak_thres] = 0.5
    config[:convnmf_binsize] = 0.5
    config[:convnmf_bkgd_thres] = 0.2
    config[:mask_data] = msk
    config[:mask_lengths] = 10.0
    config[:percent_masked] = 20.0
    config[:num_spike_resamples_per_anneal] = 10
    config[:samples_per_resample] = 10
    config[:num_spike_resamples_after_anneal] = 100

    load_train_save(config)
end

