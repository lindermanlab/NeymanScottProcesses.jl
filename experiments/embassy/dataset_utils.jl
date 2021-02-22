using DataFrames
using Dates
using NeymanScottProcesses
using Random
using SparseArrays

import CSV
import JLD

using NeymanScottProcesses: volume




# ===
# CONSTANTS
# ===

const DOCUMENT_LABEL_FILE = "raw/meta.tsv"
const VOCAB_LABEL_FILE = "raw/unique_words.txt"
const ENTITY_LABEL_FILE = "raw/unique_entities.txt"
const DATE_LABEL_FILE = "raw/dates.tsv"

const RAW_TRAIN_FILE = "raw/train.tsv"
const CLEANED_TRAIN_FILE = "intermediary/train.jld"
const CLEANED_WORD_DISTRIBUTION_FILE = "intermediary/word_distribution.jld"

DATASET_START_DATE = Date(1973, 1, 1)
DATASET_STOP_DATE = Date(1978, 12, 31)
BICENTENNIAL_DATE = Date(1976, 7, 4)





# ===
# MAIN SCRIPT
# ===

function load_train_save(datadir, config)
    C = config

    # Parse dataset parameters
    max_date = C[:max_date]
    min_date = C[:min_date]
    vocab_cutoff = C[:vocab_cutoff]

    @assert max_date > min_date

    # Load data
    data, embassy_dim, word_dim = construct_cables(datadir, min_date, max_date, vocab_cutoff)
    word_distr = load_empirical_word_distribution(datadir)
    meta = load_cables_metadata(datadir)
   
    @show length(data) 
    @show embassy_dim word_dim

    normalized_word_distr = word_distr ./ sum(word_distr, dims=1)

    # Parse model parameters
    T_max = float(max_date - min_date)
    max_cluster_radius = C[:max_cluster_radius]
    K = C[:cluster_rate]
    Ak = C[:cluster_amplitude]
    σ0 = C[:cluster_width]
    ϵ_conc = ones(embassy_dim)
    ν_conc = 1.0
    A0 = C[:background_amplitude]
    ϵ0_conc = ones(embassy_dim)  # Background embassy concentration
    ν0_conc = C[:background_word_concentration] * normalized_word_distr
    ν0_conc .+= C[:background_word_spread]
    δ_conc = ones(7)

    # Parse sampler parameters
    seed = C[:seed]
    samples_per_anneal = C[:samples_per_anneal]
    save_interval = C[:save_interval]
    temps = C[:temps]
    results_path = C[:results_path]

    # Construct model
    Random.seed!(seed)
    priors = CablesPriors(K, Ak, σ0, ϵ_conc, ν_conc, A0, ϵ0_conc, ν0_conc, δ_conc)
    model = CablesModel(T_max, priors; max_radius=max_cluster_radius)
  
    # Run sampler
    base_sampler = GibbsSampler(
        num_samples=samples_per_anneal,
        save_interval=save_interval,
        save_keys=(:log_p, :assignments)
    )

    anneal_rule(P, T) = cables_annealer(P, T, maximum(temps))
    sampler = Annealer(true, temps, anneal_rule, base_sampler)

    @time results = sampler(model, data)

    # Save results
    f = joinpath(datadir, "results", results_path)
    JLD.@save f results model

    return results, model
end

function cables_annealer(priors::CablesPriors, T, max_temp)
    μ, σ2 = mean(priors.cluster_amplitude), var(priors.cluster_amplitude)

    μ̃ = 1 + (1/log10(max_temp)) * (log10(max_temp) - log10(T)) * (μ - 1)
    σ̃2 = σ2 * (μ̃ / μ)

    priors.cluster_amplitude = specify_gamma(μ̃, σ̃2)
    return priors
end

function load_results(datadir, config)
    results_path = config[:results_path]
    min_date, max_date, vocab_cutoff = config[:min_date], config[:max_date], config[:vocab_cutoff]

    # Load data
    data, embassy_dim, word_dim = 
        construct_cables(datadir, min_date, max_date, vocab_cutoff)
    word_distr = load_empirical_word_distribution(datadir)
    meta = load_cables_metadata(datadir)

    # Load results
    r = JLD.load(joinpath(datadir, "results", results_path))

    return data, word_distr, meta, r["results"], r["model"]
end



# ===
# METADATA PROCESSING
# ===

get_dateid(date::Date, reference::Date=DATASET_START_DATE) =
    (date - reference).value

get_date(dateid::Int, reference::Date=DATASET_START_DATE) =
    reference + Day(dateid)

function load_cables_metadata(datadir)
    vocab = DataFrame(CSV.File(joinpath(datadir, VOCAB_LABEL_FILE), header=[:word])) 
    embassies = DataFrame(CSV.File(joinpath(datadir, ENTITY_LABEL_FILE), header=[:embassy]))

    return (vocab=vocab, embassies=embassies) 
end




# ===
# ANALYSIS
# ===

function inspect(cable::Cable, start_date, meta; num_words=5, word_distr=nothing)
    dateid = cable.position + start_date
    date = get_date(dateid)

    embassy = meta.embassies[cable.embassy, :embassy]

    word_scores = cable.words
    if word_distr !== nothing
        word_scores ./= word_distr[:, cable.embassy]
        word_scores .+= 1/size(word_distr, 1)  # Regularize via Laplace smoothing
    end
    word_ids = sortperm(word_scores, rev=true)[1:num_words]
    relevant_words = join(meta.vocab[word_ids, :word], " -- ")

    @info "cable details:" dateid date embassy relevant_words
    return nothing
end

function inspect(
    cluster::CableCluster, start_date, meta;
    num_words=5, num_embassies=5, word_distr=nothing
)
    e = cluster.sampled_embassy_probs
    ν = cluster.sampled_word_probs
    
    dateid = round(Int, cluster.sampled_position) + get_dateid(start_date)
    date = get_date(dateid)

    embassy_ids = sortperm(e, rev=true)[1:num_embassies]
    relevant_embassies = join(meta.embassies[embassy_ids, :embassy], " -- ")

    word_scores = ν
    if word_distr !== nothing
        V0 = word_distr
        n = size(word_distr, 1)
        word_scores ./= (V0*e .+ 1/n) 
    end
    word_ids = sortperm(word_scores, rev=true)[1:num_words]
    relevant_words = join(meta.vocab[word_ids, :word], " -- ")

    @info "cluster details:" dateid date relevant_embassies relevant_words
    return nothing
end

function get_cluster_stats(model)
    f(c) = (μ=c.sampled_position, σ=sqrt(c.sampled_variance), A=c.sampled_amplitude)
    stats = Any[(-1, (μ=nothing, σ=nothing, A=model.globals.bkgd_rate * volume(model)))]
    for i in model.clusters.indices
        c = model.clusters[i]
        push!(stats, (i, f(c)))
    end

    return stats
end

function get_empirical_cluster_stats(data, assignments)
    cluster_ids = sort(unique(assignments))
    stats = []
    for i in cluster_ids
        cluster_data = data[assignments .== i]
        times = [x.position for x in cluster_data]

        μ = mean(times)
        σ2 = mean(times .^ 2) - μ^2
        A = length(times)

        push!(stats, (i, (μ=μ, σ=sqrt(σ2), A=A)))
    end

    return stats
end




# ===
# PREPROCESSING
# ===

function construct_cables(datadir, min_date, max_date, vocab_cutoff)
    # Load data
    docs, dates, embassies, doc_word_mat = load_input_data(datadir)
    word_distr = load_empirical_word_distribution(datadir)

    # Truncate vocabulary if needed
    truncate_vocab!(doc_word_mat, word_distr, vocab_cutoff)

    # Filter nodes by date
    docs, dates, embassies, doc_word_mat = 
        filter_by_date(docs, dates, embassies, doc_word_mat, min_date, max_date)

    # Construct cables
    _make_cable(i) = Cable(dates[i], embassies[i], doc_word_mat[:, i])
    data = _make_cable.(1:length(docs))

    # Remove cables with zero words, which occur due to vocab truncation
    filter!(x -> x._word_sum > 0, data)

    num_words, num_embassies = size(word_distr)

    return data, num_embassies, num_words
end

function truncate_vocab!(doc_word_mat, word_distr, vocab_cutoff)
    word_counts = sum(word_distr, dims=2)[:, 1]
    top_words = sortperm(word_counts, rev=true)[1:vocab_cutoff]
    
    doc_word_mat[top_words, :] .= 0
    word_distr[top_words, :] .= 0

    return doc_word_mat, word_distr
end

function filter_by_date(docs, dates, embassies, doc_word_mat, min_date, max_date)
    relevant = (min_date .<= dates .<= max_date)

    docs = docs[relevant]
    dates = dates[relevant] .- min_date
    embassies = embassies[relevant]
    doc_word_mat = doc_word_mat[:, docs]

    return docs, dates, embassies, doc_word_mat
end

function load_input_data(datadir; refresh=false)
    f = joinpath(datadir, CLEANED_TRAIN_FILE)
    if isfile(f) && (!refresh)
        d = JLD.load(f)
        docs = d["docs"]
        dates = float.(d["dates"])
        embassies = d["embassies"]
        doc_word_mat = d["doc_word_mat"]
        return docs, dates, embassies, doc_word_mat

    else
        @info "Running preprocessing script."
        return preprocess_input_data(datadir)
    end
end

function load_empirical_word_distribution(datadir; refresh=false)
    f = joinpath(datadir, CLEANED_WORD_DISTRIBUTION_FILE)
    if isfile(f) && (!refresh)
        d = JLD.load(f)
        word_distr = d["word_distr"]
        return word_distr
    else
        @info "Running preprocessing script."
        return preprocess_empirical_word_distribution(datadir)
    end
end

function preprocess_input_data(datadir)
    # Generate the document word matrix
    doc_inds = Int[]
    word_inds = Int[]
    word_counts = Int[]
    for row in CSV.Rows(
        joinpath(datadir, RAW_TRAIN_FILE), 
        header=[:docid, :wordid, :word_count],
        types=[Int, Int, Int]
    )
        push!(doc_inds, row.docid+1)  # In Julia, everything will be 1-indexed
        push!(word_inds, row.wordid+1)
        push!(word_counts, row.word_count)
    end
    doc_word_mat = sparse(word_inds, doc_inds, word_counts)

    # Clean up garbage to prevent an OOM error
    word_inds, word_counts = nothing, nothing
    GC.gc()

    # Identify unique documents
    docs = unique(doc_inds)

    # Load the embassies and dates of each document
    document_metadata = DataFrame(CSV.File(
        joinpath(datadir, DOCUMENT_LABEL_FILE), header=[:docid, :embassy, :date]
    ))
    embassies = document_metadata[docs, :embassy] .+ 1
    dates = document_metadata[docs, :date]

    # Save results
    f = joinpath(datadir, CLEANED_TRAIN_FILE)
    JLD.@save f docs dates embassies doc_word_mat

    return docs, dates, embassies, doc_word_mat
end

function preprocess_empirical_word_distribution(datadir)
    docs, dates, embassies, doc_word_mat = load_input_data(datadir)
    word_distr = compute_empirical_word_distribution(docs, embassies, doc_word_mat)

    f = joinpath(datadir, CLEANED_WORD_DISTRIBUTION_FILE)
    JLD.@save f word_distr

    return word_distr
end




# ===
# COMPUTING EMPIRICAL STATISTICS
# ===

function compute_empirical_word_distribution(docs, embassies, doc_word_mat)
    num_docs = length(docs)
    num_words = size(doc_word_mat, 1)
    num_embassies = length(unique(embassies))

    word_distr = zeros(num_words, num_embassies)
    for i in 1:num_docs
        word_distr[:, embassies[i]] .+= doc_word_mat[:, docs[i]]
    end

    return word_distr
end

function compute_empirical_embassy_distribution(embassies)
    return [count(ϵ -> (ϵ == k), embassies) for k in 1:maximum(embassies)]
end


