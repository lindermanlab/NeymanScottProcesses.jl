get_jobpath(dataset_id::String, sweep_id::Int, job_id::Int) = (
    joinpath(
        pwd(),
        "results",
        dataset_id,
        @sprintf("%03i", sweep_id),
        @sprintf("%05i", job_id),
    )
)

get_sweeppath(dataset_id::String, sweep_id::Int) = (
    joinpath(
        pwd(),
        "results",
        dataset_id,
        @sprintf("%03i", sweep_id),
    )
)

"""
Loads neural dataset.

Params
------
dataset_id : String, options are: ("songbird")

Returns
-------
spikes : Vector{Spike}
max_time : Float64
num_neurons : Int64
"""
function load_dataset(dataset_id)


    # => Songbird data collected by E. Mackevicius and M. Fee
    #    at MIT. Available at: https://github.com/FeeLab/seqNMF
    if dataset_id == "songbird"

        file = MAT.matopen(
            joinpath(DATAPATH, "songbird/MackeviciusData.mat"))
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

        # Permute neuron ids so sequences are visible.
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

    
    # => Synthetic spike train with high SNR.
    elseif dataset_id == "synthetic"
        
        BSON.@load joinpath(DATAPATH, "synthetic/data.bson") spikes
        config = YAML.load(open(joinpath(DATAPATH, "synthetic/config.yml")))
        max_time = config["max_time"]
        num_neurons = config["num_neurons"]

    # => Synthetic spike train with warping.
    elseif dataset_id == "warped"
        
        BSON.@load joinpath(DATAPATH, "warped/data.bson") spikes
        config = YAML.load(open(joinpath(DATAPATH, "warped/config.yml")))
        max_time = config["max_time"]
        num_neurons = config["num_neurons"]

    # => Hippocampal data
    elseif dataset_id == "hippocampus"

        BSON.@load joinpath(DATAPATH, "hippocampus/maze_spikes.bson") maze_spikes
        BSON.@load joinpath(DATAPATH, "hippocampus/maze_metadata.bson") max_time num_neurons
        spikes = maze_spikes

    # => Hippocampal data
    elseif dataset_id == "big_maze"

        BSON.@load joinpath(DATAPATH, "hippocampus/big_maze_spikes.bson") big_maze_spikes
        BSON.@load joinpath(DATAPATH, "hippocampus/big_maze_metadata.bson") max_time num_neurons
        spikes = big_maze_spikes

    else
        throw(ArgumentError(
            @sprintf("dataset \"%s\" not recognized.", dataset_id)
        ))
    end

    return spikes, max_time, num_neurons
end


load_num_neurons(dataset_id::String, sweep_id::Int64, job_id::Int64) = (
    PointProcessSequences.num_neurons(load_model(dataset_id, sweep_id, job_id))
)
load_num_sequence_types(dataset_id::String, sweep_id::Int64, job_id::Int64) = (
    PointProcessSequences.num_sequence_types(load_model(dataset_id, sweep_id, job_id))
)

function load_model(dataset_id::String, sweep_id::Int64, job_id::Int64)

   jobpath = get_jobpath(
        dataset_id,
        sweep_id,
        job_id
    )
    BSON.@load joinpath(jobpath, "model.bson") model

    return model

end

function load_init_model(dataset_id::String, sweep_id::Int64, job_id::Int64)

   jobpath = get_jobpath(
        dataset_id,
        sweep_id,
        job_id
    )
    BSON.@load joinpath(jobpath, "initial_model.bson") initial_model

    return initial_model

end


function load_config(dataset_id::String, sweep_id::Int64, job_id::Int64)
    jobpath = get_jobpath(
        dataset_id,
        sweep_id,
        job_id
    )
    return Dict(
        Symbol(k) => v for (k, v) in YAML.load(open(joinpath(jobpath, "config.yml")))
    )
end

function load_results(dataset_id::String, sweep_id::Int64, job_id::Int64)
    jobpath = get_jobpath(
        dataset_id,
        sweep_id,
        job_id
    )
    BSON.load(joinpath(jobpath, "results.bson"))
end

function load_masks(dataset_id::String, sweep_id::Int64, job_id::Int64)
    try
        load_results(dataset_id, sweep_id, job_id)[:masks]
    catch e
        return Mask[]
    end
end
