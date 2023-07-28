const NOT_SAMPLED_AMPLITUDE = -1.0


function Cluster()
    return Cluster(
        0,
        (zeros(2), zeros(2, 2)),
        zeros(2),
        zeros(2, 2),
        NOT_SAMPLED_AMPLITUDE
    )
end


constructor_args(e::Cluster) = ()


function reset!(e::Cluster)
    e.datapoint_count = 0
    e.moments[1] .= 0
    e.moments[2] .= 0
end


function been_sampled(e::Cluster)
    return !isapprox(amplitude(e), NOT_SAMPLED_AMPLITUDE)
end


function too_far(d::Point, e::Cluster, m::GaussianNeymanScottModel)
    return euclidean_distance(position(d), position(e)) > m.max_event_radius
end


function remove_datapoint!(
    model::GaussianNeymanScottModel, 
    x::Point, 
    k::Int;
    recompute_posterior::Bool=true
) 
    e = events(model)[k]

    # If this is the last spike in the event, we can return early.
    (e.datapoint_count == 1) && (return remove_event!(events(model), k))


    e.datapoint_count -= 1
    e.moments[1] .-= position(x)

    # Unroll second moment (Σ += xx') update for performance
    for i = 1:2
        for j = 1:2
            e.moments[2][i, j] -= position(x)[i] * position(x)[j]
        end
    end

    if recompute_posterior
        set_posterior!(model, k)
    end
end


function add_datapoint!(
    model::GaussianNeymanScottModel, 
    x::Point, 
    k::Int;
    recompute_posterior::Bool=true
) 
    e = events(model)[k]

    e.datapoint_count += 1
    e.moments[1] .+= position(x)

    # Unroll second moment (Σ += xx') update for performance
    for i = 1:2
        for j = 1:2
            e.moments[2][i, j] += position(x)[i] * position(x)[j]
        end
    end

    if recompute_posterior
        set_posterior!(model, k)
    end

    return k
end


function event_list_summary(m::GaussianNeymanScottModel)
    ev = events(m)
    return [
        GaussianEventSummary((
            ind,
            ev[ind].sampled_amplitude,
            ev[ind].sampled_position, 
            ev[ind].sampled_covariance
        )) 
        for ind in ev.indices
    ]
end


"""Use this function to do memory-free Euclidean distance."""
function euclidean_distance(x, y)
    return sqrt(mapreduce((x, y) -> (x+y)^2, +, x, y))
end