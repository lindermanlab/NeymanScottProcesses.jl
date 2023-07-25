import Base.Iterators

# Loads PPSeq module, and file IO wrappers.
include("./utils/all_utils.jl")

# Load ground truth config.
config = Dict{Symbol,Union{Int64,Float64,String}}(
    Symbol(key) => val for (key, val) in YAML.load(open(joinpath(DATAPATH, "warped", "config.yml")))
)

# Generate configs
random_seeds = (1, 2, 3)
num_sequence_types = (2,)
max_warps = (3.0,)
mask_strategies = ("none",)
# mask_strategies = ("none", "randomly", "in blocks")

itr = Iterators.product(
    random_seeds,
    num_sequence_types,
    max_warps,
    mask_strategies
)

sweep_id = 4
@printf("[ time: %s ]\nStarting sweep %i\n", Dates.now(), sweep_id)
@printf(" WORKER ID : %i", parse(Int64, ENV["SLURM_ARRAY_TASK_ID"]))
@printf("===============================\n", )

job_id = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])
(seed, R, mx_wrp, msk) = collect(itr)[job_id]
    
@printf("===============================\n", )
@printf("[ time: %s ]\n", Dates.now())
@printf("Sweep %i, Running job %i / %i\n", sweep_id, job_id, length(itr))

# Update config.
config[:sweep_id] = sweep_id
config[:job_id] = job_id
config[:seed] = seed
config[:num_sequence_types] = R
config[:max_warp] = mx_wrp
config[:num_warp_values] = 15
config[:num_threads] = 0

# Add details to config about inference.
config[:num_anneals] = 10
config[:samples_per_anneal] = 100
config[:max_temperature] = 1e4
config[:save_every_during_anneal] = 1
config[:samples_after_anneal] = 100
config[:save_every_after_anneal] = 1
config[:split_merge_moves_during_anneal] = 0
config[:split_merge_moves_after_anneal] = 0
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

