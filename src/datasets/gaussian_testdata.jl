

function gaussian_2d_data(;
        domain = Box([4, 4]),
        cov_scale = 5e-3 * I(2),
        cov_df = 5.0,
        mean_prior = [b/2 for b in domain.bounds],
        mean_pseudo_obs = 0.01,
        clus_rate = 4.0,
        clus_amp_dist = specify_gamma(20.0, 3.0),
        bkgd_amp_dist = specify_gamma(20.0, 3.0)
    )

    # Specify priors for mixture components.
    cluster_priors = GaussianPriors(
        cov_scale,
        cov_df,
        mean_prior,
        mean_pseudo_obs
    )

    # Specify priors for Neyman-Scott process.
    model_priors = NeymanScottPriors(
        clus_rate,      # rate parameter for cluster intensity function.
        clus_amp_dist,  # gamma distribution on cluster amp.
        bkgd_amp_dist,  # gamma distribution on background amp.
        cluster_priors  # normal-inverse-wishart
    )

    # Instantiate model
    model = NeymanScottModel(
        domain,
        model_priors
    )

    # Sample the model
    datapoints, true_assignments, true_clusters = sample_full_process(model)

    return model, datapoints, true_assignments, true_clusters

end

function masked_gaussian_2d_data(;
        mask_radius = 0.1,
        num_heldout_regions = 300,
        kwargs...
    )
    

    # Construct the model and sample dataset.
    (
        model,
        data,
        assgn,
        true_clusters
    ) = gaussian_2d_data(kwargs...)


    # Sample heldout_region.
    heldout_region = sample_random_spheres(
        model.domain,
        num_heldout_regions,
        mask_radius
    )

    # Split data and assignments into oberved and heldout regions.
    heldout_idx = [(x in heldout_region) for x in data]
    observed_idx = .!(heldout_idx)

    heldout_data = data[heldout_idx]
    observed_data = data[observed_idx]

    heldout_assignments = assgn[heldout_idx]
    observed_assignments = assgn[observed_idx]

    # Return model and split assignments.
    return (
        model,
        heldout_region,
        heldout_data,
        observed_data,
        heldout_assignments,
        observed_assignments,
        true_clusters
    )
end