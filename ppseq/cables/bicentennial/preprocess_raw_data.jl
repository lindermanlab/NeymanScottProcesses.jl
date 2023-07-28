import CSV
import JLD
using DataFrames
using SparseArrays

DATADIR = "/Users/degleris/data/cables/"

UNIQUE_ENTITIES = "unique_entities.txt"
UNIQUE_DATES = "unique_dates.txt"
META = "meta.tsv"
DATES = "dates.tsv"
TRAIN = "train.tsv"
TEST = "test.tsv"
VALIDATION = "validation.tsv"

dataset_name = TRAIN

# Get the sparse doc-word matrix
# Instead of using a DataFrame, we'll have
# to do this in place because of the size of
# the train dataset.
doc_inds = Int[]
word_inds = Int[]
word_counts = Int[]
for row in CSV.Rows(
    DATADIR * dataset_name, 
    header=[:docid, :wordid, :word_count],
    types=[Int, Int, Int]
)
    push!(doc_inds, row.docid+1)
    push!(word_inds, row.wordid+1)
    push!(word_counts, row.word_count)
end
doc_word_mat = sparse(word_inds, doc_inds, word_counts)

word_inds, word_counts = nothing, nothing  # Clean garbage
GC.gc()

# Get the nodes and dates for each document
meta = DataFrame(CSV.File(DATADIR * META, header=[:docid, :nodeid, :date]))
docs = unique(doc_inds)
nodes = meta[docs, :nodeid]
dates = meta[docs, :date]

# Save results
filepath = DATADIR * dataset_name[1 : end-4] * ".jld"
JLD.@save filepath docs dates nodes doc_word_mat
