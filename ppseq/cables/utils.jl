using CSV
using DataFrames
using Dates
using Distributions
using PointProcessSequences
using SparseArrays
using Random

using JLD: load, @save
using PointProcessSequences: CableCluster, event_list_summary

TRAIN = "train.jld"
DATASET_START_DATE = Date(1973, 1, 1)
DATASET_STOP_DATE = Date(1978, 12, 31)

function load_train_save(datadir, config)
    # Load data
    total_word_distr = load_total_word_distribution(datadir)
    data, num_nodes, num_words = load_cables_dataset(
        datadir, config[:min_date], config[:max_date];
        vocab_cutoff=config[:vocab_offset], total_word_distr=total_word_distr
    )
    vocab, embassies, dates = load_cables_metadata(datadir)
   
    @show length(data)
    @show num_words
    @show num_nodes

    # Train model
    Random.seed!(config[:seed])
    
    max_time = float(config[:max_date] - config[:min_date])
    node_concentration = config[:node_concentration] * ones(num_nodes)
    mark_concentration = config[:mark_concentration]
    bkgd_node_concentration = config[:bkgd_node_concentration] * ones(num_nodes)
    bkgd_mark_concentration = config[:bkgd_mark_strength] * total_word_distr
    bkgd_mark_concentration .+= config[:bkgd_mark_spread]

    priors = CablesPriors(
        config[:event_rate],
        config[:event_amplitude],
        config[:bkgd_amplitude],
        node_concentration,
        mark_concentration,
        bkgd_node_concentration,
        bkgd_mark_concentration,
        config[:variance_scale],
        config[:variance_pseudo_obs],
        Dirichlet(config[:bkgd_day_of_week]),
    )

    primary_model = CablesModel(max_time, config[:max_radius], priors)
    if config[:num_partitions] == 1
        model = primary_model
    else
        model = make_distributed(primary_model, config[:num_partitions])
    end

    (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) = annealed_gibbs!(
        model, data, fill(-1, length(data));
        num_anneals=config[:num_anneals],
        samples_per_anneal=config[:samples_per_anneal],
        max_temperature=config[:max_temp],
        save_every=config[:save_every],
        verbose=true,
        save_set=[:assignments]
    )

    # Save results
    results = Dict(
        :config => config,
        :model => primary_model,
        :assignments => assignments,
        :events => event_list_summary(primary_model),
        :assignment_hist => assignment_hist,
        :log_p_hist => log_p_hist,
    )
    
    results_path = datadir * config[:outfile]
    @save results_path results
    return results
end


function load_cables_dataset(
    datadir, min_date, max_date; 
    vocab_cutoff=0, total_word_distr=nothing
)
    if vocab_cutoff > 0
        @assert total_word_distr !== nothing
    end

    d = load(datadir * TRAIN)
    docs = d["docs"]
    dates = float.(d["dates"])
    nodes = d["nodes"] .+ 1
    doc_word_mat = d["doc_word_mat"]
    d = nothing
    GC.gc()

    num_words = size(doc_word_mat, 1)
    num_nodes = maximum(nodes)
    
    # Truncate vocab
    if vocab_cutoff > 0
        word_count = sum(total_word_distr, dims=1) 
        word_freq = sortperm(word_count[1, :], rev=true)
        common_words = word_freq[1:vocab_cutoff]
    
        total_word_distr[common_words, :] .= 0
        doc_word_mat[common_words, :] .= 0
    end

    # Filter nodes by date
    relevant_docs = (min_date .<= dates .<= max_date)

    docs = docs[relevant_docs]
    dates = dates[relevant_docs] .- min_date
    nodes = nodes[relevant_docs]
    doc_word_mat = doc_word_mat[:, docs]
    
    GC.gc()

    # Finally, convert all these to cables
    println("Converting raw data to `Cable`s...")
    _make_cable(i) = Cable(dates[i], nodes[i], doc_word_mat[:, i])
    data = _make_cable.(1:length(docs))

    # Remove cables with zero words (this occurs because of vocab truncation)
    filter!(x -> x._mark_sum > 0, data)

    return data, num_nodes, num_words 
end

function load_cables_metadata(datadir)
    vocab = DataFrame(CSV.File(datadir * "unique_words.txt", header=[:word])) 
    embassies = DataFrame(CSV.File(datadir * "unique_entities.txt", header=[:node]))
    dates = DataFrame(CSV.File(datadir * "dates.tsv"))

    return vocab, embassies, dates 
end

function load_total_word_distribution(datadir)
    d = load(datadir * "empirical_word_distr.jld")
    return d["word_distr"]
end

function inspect(
    cable::Cable, min_date, vocab, embassies, dates;
    num_words=5, word_distr=nothing
)
    if word_distr === nothing
        mark = cable.mark
    else
        mark = cable.mark ./ (word_distr[:, cable.node] .+ 1/size(word_distr, 1))
    end

    dateid = cable.position + min_date

    date = dates[dates[:id] .== dateid, :date][1]
    embassy = embassies[cable.node, :node]
    word_inds = sortperm(mark, rev=true)[1:num_words]
    words = join(vocab[word_inds, :word], " -- ")

    @info "cable details:" dateid date embassy words
    return nothing
end
function inspect(
    cluster::CablesEventSummary, min_date, vocab, embassies, dates;
    num_words=5, num_embassies=5, word_distr=nothing
)
    if word_distr === nothing
        mark = cluster.markproba
    else
        mark = cluster.markproba ./ (word_distr*cluster.nodeproba .+ 1/size(word_distr, 1)) 
    end

    dateid = round(Int, cluster.position) + min_date
    
    embassy_inds = sortperm(cluster.nodeproba, rev=true)[1:num_embassies]
    word_inds = sortperm(mark, rev=true)[1:num_words]
    
    date = dates[dates[:id] .== dateid, :date][1]
    relevant_embassies = join(embassies[embassy_inds, :node], " -- ")
    relevant_words = join(vocab[word_inds, :word], " -- ")

    @info "cluster details:" dateid date relevant_embassies relevant_words
    return nothing
end

function empirical_node_distribution(data::Vector{Cable}, num_nodes)
    empirical_distr = zeros(num_nodes)

    for cable in data
        empirical_distr[cable.node] += 1
    end

    return empirical_distr / sum(empirical_distr)
end

function empirical_word_distribution(data::Vector{Cable}, num_nodes, num_words)
    empirical_distr = zeros(num_words, num_nodes)

    for x in data
        empirical_distr[:, x.node] .+= (x.mark / (sum(x.mark) .+ eps()))
    end

    normalization_constants = sum(empirical_distr, dims=1) .+ eps()
    return empirical_distr ./ normalization_constants
end

function get_dateid(date::Date, reference::Date=DATASET_START_DATE)
    return (date - reference).value
end
