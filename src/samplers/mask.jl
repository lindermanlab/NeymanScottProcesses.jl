struct MaskedSampler{M <: AbstractMask} <: AbstractSampler
    verbose::Bool
    num_samples::Int
    subsampler::AbstractSampler
    masks::Vector{M}
    masked_data::Union{Vector{<: AbstractDatapoint}, Nothing}
end

valid_save_keys(S::MaskedSampler) = (:train_log_p, :test_log_p, valid_save_keys(subsampler)...)

MaskedSampler(subsampler, num_samples, masks; verbose=true, masked_data=nothing) =
    MaskedSampler(verbose, num_samples, subsampler, masks, masked_data)

function Base.getproperty(obj::MaskedSampler, sym::Symbol)
    if sym === :save_interval
        return 1
    elseif sym === :save_keys
        return (:train_log_p, :test_log_p, subsampler.save_keys...)
    else
        return getfield(obj, sym)
    end
end

function (S::MaskedSampler)(
    model::NeymanScottModel,
    unmasked_data::Vector{T};
    initial_assignments::Vector{Int64}=:background,
) where {T <: AbstractDatapoint}
    verbose, num_samples = S.verbose, S.num_samples
    subsampler, masked_data, masks = S.subsampler, S.masked_data, S.masks

    # Compute inverse masks
    inv_masks = compute_complementary_masks(masks, model)

    # Sanity check.
    @assert length(unmasked_data) === length(initial_assignments)
    _check_masking_sanity(unmasked_data, masked_data, masks, inv_masks)

    # Compute relative masked volume and baselines
    pc_masked = 1 - masked_proportion(model, inv_masks)
    train_baseline = baseline_log_like(unmasked_data, inv_masks)
    test_baseline = (masked_data !== nothing) ? baseline_log_like(masked_data, masks) : 0.0

    # Initialize assignments, results, and sampled data
    unmasked_assignments = initialize_assignments(unmasked_data, initial_assignments)
    results = initialize_results(model, unmasked_assignments, S)
    sampled_data, sampled_assignments = T[], Int64[]

    verbose && @show pc_masked train_baseline test_baseline

    for i in 1:num_samples
        # Sample fake data in masked regions and run subsampler
        sample_masked_data!(sampled_data, sampled_assignments, model, masks)
        _data = vcat(unmasked_data, sampled_data)
        _assgn = vcat(unmasked_assignments, sampled_assignments)

        # Run sampler
        new_results = subsampler(model, _data, _assgn)

        # Update assignments
        unmasked_assignments .= view(last(new_results.assignments), 1:length(unmasked_data))

        # Update results
        append_results!(results, new_results, subsampler)
        push!(results.train_log_p, log_like(model, unmasked_data, inv_masks)) - train_baseline
        
        # If masked data is available as well, compute test log likelihood
        if masked_data !== nothing
            push!(results.test_log_p, log_like(model, masked_data, masks) - test_baseline)
        else
            push!(results.test_log_p, 0.0)
        end

        verbose && print(i * samples_per_resample, "-")
    end
    verbose && println("Done")

    # Before returning, remove assignments assigned to imputed spikes.
    recompute_statistics!(model, unmasked_data, unmasked_assignments)
    
    # TODO Rescale likelihoods?
   
    return results
end

function _check_masking_sanity(unmasked_data, masked_data, masks, inv_masks)
    @assert all([x ∈ inv_masks for x in unmasked_data])
    @assert all([x ∉ masks for x in unmasked_data])

    if masked_data !== nothing
        @assert all([x ∈ masks for x in masked_data])
        @assert all([x ∉ inv_masks for x in masked_data])
    end
end
