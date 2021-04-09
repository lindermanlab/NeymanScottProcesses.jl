"""
    log_like(model, data)

Log likelihood of `data` given `model`.
"""
function log_like(model::NeymanScottModel, data::Vector)

    ll = 0.0
    for x in data
        g = log_bkgd_intensity(model, x)

        for cluster in model.cluster_list
            g = logaddexp(g, log_cluster_intensity(model, cluster, x))
        end

        ll += g
    end 
    
    ll -= model.globals.bkgd_rate * volume(model.domain)
    for cluster in model.cluster_list
        ll -= cluster.sampled_amplitude
    end
    
    return ll
end


"""
    log_like(model, data, mask::Region)

Log likelihood of `data` within a sub-region `mask`.
"""
function log_like(
    model::NeymanScottModel,
    data::AbstractVector,
    mask::Region
)

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all datapoints -- #
    ll = 0.0
    for x in data
        ll += log_bkgd_intensity(model, x)
        for cluster in model.cluster_list
            logaddexp(ll, log_cluster_intensity(model, cluster, x))
        end
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #
    ll -= integrated_bkgd_intensity(model, mask)
    for cluster in model.cluster_list
        ll -= integrated_cluster_intensity(model, cluster, mask)
    end

    return ll
end


