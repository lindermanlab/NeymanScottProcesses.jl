struct MaskedSampler <: AbstractSampler
    verbose::Bool
    num_samples::Int
    subsampler::AbstractSampler
    heldout_region::AbstractMask
    heldout_data::Union{Vector{<: AbstractDatapoint}, Nothing}
end

valid_save_keys(S::MaskedSampler) = (:train_log_p, :test_log_p, valid_save_keys(S.subsampler)...)

MaskedSampler(subsampler, masks; verbose=true, num_samples=10, heldout_data=nothing) =
    MaskedSampler(verbose, num_samples, subsampler, masks, heldout_data)

function Base.getproperty(obj::MaskedSampler, sym::Symbol)
    if sym === :save_interval
        return 1
    elseif sym === :save_keys
        return (:train_log_p, :test_log_p, obj.subsampler.save_keys...)
    else
        return getfield(obj, sym)
    end
end

function (S::MaskedSampler)(
    model::NeymanScottModel,
    observed_data::Vector{T};
    initial_assignments::Vector{Int64}=:background,
) where {T <: AbstractDatapoint}

    # Grab sampler parameters.
    verbose = S.verbose
    num_samples = S.num_samples
    subsampler = S.subsampler
    heldout_data = S.heldout_data
    heldout_region = S.heldout_region

    # Compute complement of masked region.
    observed_region = ComplementMask(heldout_region, volume(model))

    # Sanity checks.
    @assert length(observed_data) === length(initial_assignments)
    @assert all_data_in_masks(observed_data, observed_region)
    @assert all_data_in_masks(heldout_data, heldout_region)
    @assert all_data_not_in_masks(observed_data, heldout_region)
    @assert all_data_not_in_masks(heldout_data, observed_region)

    # Compute relative masked volume and baselines
    percent_heldout = 100 * volume(heldout_region) / volume(model)
    train_baseline = baseline_log_like(observed_data, observed_region)
    test_baseline = (heldout_data === nothing) ? 0.0 : baseline_log_like(heldout_data, heldout_region)

    # Initialize assignments, results, and sampled data
    assignments = initialize_assignments(observed_data, initial_assignments)
    results = initialize_results(model, assignments, S)

    # Allocate vectors for imputed datapoints.
    imputed_data = T[]
    imputed_assignments = Int64[]

    # Display statistics of interest.
    verbose && @show percent_heldout train_baseline test_baseline

    for i in 1:num_samples

        # Sample fake data in masked region and run subsampler
        sample_data_in_mask!(
            imputed_data,
            imputed_assignments,
            model,
            heldout_region
        )

        # Concatenate imputed data and assignments.
        _data = vcat(observed_data, imputed_data)
        _assgn = vcat(assignments, imputed_assignments)

        # Run sampler on observed and imputed data.
        new_results = subsampler(model, _data; initial_assignments=_assgn)
        assignments .= view(last(new_results.assignments), 1:length(observed_data))

        # Update results ()
        update_results!(
            results, model, assignments, observed_data, S
        )

        # Compute log-likelihood on "train set"
        push!(
            results.train_log_p,
            normalized_log_like(model, observed_data, observed_region, train_baseline)
        )

        # Compute log-likelihood on "test set"
        push!(
            results.test_log_p,
            normalized_log_like(model, heldout_data, heldout_region, test_baseline)
        )

    end

    # Before returning, remove assignments assigned to imputed spikes.
    recompute_cluster_statistics!(model, observed_data, assignments)
    
    # TODO Rescale likelihoods?
   
    return results
end

all_data_in_masks(data, masks) = (data === nothing) || all([x ∈ masks for x in data])

all_data_not_in_masks(data, masks) = (data === nothing) || all([x ∉ masks for x in data])

normalized_log_like(model, data, masks, baseline) =
    (data == nothing) ? 0.0 : log_like(model, data, masks) - baseline
