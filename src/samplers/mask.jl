struct MaskedSampler{M <: AbstractMask} <: AbstractSampler
    verbose::Bool
    num_samples::Int
    subsampler::AbstractSampler
    masks::Vector{M}
    masked_data::Union{Vector{<: AbstractDatapoint}, Nothing}
end

valid_save_keys(S::MaskedSampler) = (:train_log_p, :test_log_p, valid_save_keys(S.subsampler)...)

MaskedSampler(subsampler, masks; verbose=true, num_samples=10, masked_data=nothing) =
    MaskedSampler(verbose, num_samples, subsampler, masks, masked_data)

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
    unmasked_data::Vector{T};
    initial_assignments::Union{Vector{Int64}, Symbol}=:background,
) where {T <: AbstractDatapoint}
    verbose, num_samples = S.verbose, S.num_samples
    subsampler, masked_data, masks = S.subsampler, S.masked_data, S.masks

    # Compute inverse masks
    inv_masks = complement_masks(masks, model)

    # Sanity check.
    @assert (typeof(initial_assignments) <: Symbol) || (length(unmasked_data) === length(initial_assignments))   
    @assert all_data_in_masks(unmasked_data, inv_masks)
    @assert all_data_in_masks(masked_data, masks)
    @assert all_data_not_in_masks(unmasked_data, masks)
    @assert all_data_not_in_masks(masked_data, inv_masks)

    # Compute relative masked volume and baselines
    pc_masked = 1 - masked_proportion(model, inv_masks)
    train_baseline = baseline_log_like(unmasked_data, inv_masks)
    test_baseline = (masked_data === nothing) ? 0.0 : baseline_log_like(masked_data, masks)

    # Initialize assignments, results, and sampled data
    unmasked_assignments = initialize_assignments(unmasked_data, initial_assignments)
    results = initialize_results(model, unmasked_assignments, S)
    sampled_data, sampled_assignments = T[], Int64[]

    # verbose && @show pc_masked train_baseline test_baseline

    for i in 1:num_samples
        # Sample fake data in masked regions and run subsampler
        sample_masked_data!(sampled_data, sampled_assignments, model, masks)
        _data = vcat(unmasked_data, sampled_data)
        _assgn = vcat(unmasked_assignments, sampled_assignments)

        # Run sampler
        new_results = subsampler(model, _data; initial_assignments=_assgn)
        unmasked_assignments .= view(last(new_results.assignments), 1:length(unmasked_data))

        # Update results
        # append_results!(results, new_results, subsampler)
        update_results!(results, model, unmasked_assignments, unmasked_data, S)
        push!(results.train_log_p, normalized_log_like(model, unmasked_data, inv_masks, train_baseline))
        push!(results.test_log_p, normalized_log_like(model, masked_data, masks, test_baseline))
    end
    # Before returning, remove assignments assigned to imputed spikes.
    recompute_cluster_statistics!(model, clusters(model), unmasked_data, unmasked_assignments)
    
    # TODO Rescale likelihoods?

    verbose && println("Train log like: $(results.train_log_p[end])")
    verbose && println("Test log like: $(results.test_log_p[end])\n")
   
    return results
end

all_data_in_masks(data, masks) = (data === nothing) || all([x ∈ masks for x in data])

all_data_not_in_masks(data, masks) = (data === nothing) || all([x ∉ masks for x in data])

normalized_log_like(model, data, masks, baseline) =
    (data === nothing) ? 0.0 : log_like(model, data, masks) - baseline
