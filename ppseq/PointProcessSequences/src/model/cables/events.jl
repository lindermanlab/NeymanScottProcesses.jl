"""
Returns an empty event. May specify arguments if desired.
"""
function CableCluster(num_nodes::Int, num_marks::Int)
    return CableCluster(
        0,
        (0.0, 0.0),
        zeros(Int, num_nodes),
        zeros(Int, num_marks),

        0.0,
        0.0,
        NOT_SAMPLED_AMPLITUDE,
        zeros(num_nodes),
        zeros(num_marks)
    )   
end


"""
Returns the arguments of used to generate an empty event similar to `event`.

This is helpful when, for example, different instances of the model require
slightly different structures (for example, in a neuroscience dataset
the number of neurons will determine the size of many arrays in the event).
"""
constructor_args(event::CableCluster) = (
    length(event.node_count),
    length(event.mark_count),
)


"""
Resets the sufficient statistics and sampled values of `event`, as if it
were empty.
"""
function reset!(cluster::CableCluster)
    cluster.datapoint_count = 0.0
    cluster.node_count .= 0.0
    cluster.mark_count .= 0.0
    cluster.moments = (0.0, 0.0)
end


"""
Returns `true` if the event has already been sampled.
"""
been_sampled(cluster::CableCluster) =
    (cluster.sampled_amplitude != NOT_SAMPLED_AMPLITUDE)


"""
Removes the point `x` from event `k` in `events(model)`.
"""
function remove_datapoint!(
    model::CablesModel, 
    x::Cable, 
    k::Int64;
    recompute_posterior::Bool=true
) 

    e = events(model)[k]

    # If this is the last spike in the event, we can return early.
    (e.datapoint_count == 1) && (return remove_event!(events(model), k))


    e.datapoint_count -= 1
    e.moments = (e.moments[1] - time(x), e.moments[2] - time(x)^2)

    # Update node count
    e.node_count[x.node] -= 1
    
    # Update word count
    e.mark_count .-= x.mark

    if recompute_posterior
        set_posterior!(model, k)
    end
end

"""
Adds the point `x` to event `k` in `events(model)`.
"""
function add_datapoint!(
    model::CablesModel, 
    x::Cable, 
    k::Int64;
    recompute_posterior::Bool=true
) 
    e = events(model)[k]
    #@show e
    #@show e.moments

    e.datapoint_count += 1
    e.moments = (e.moments[1] + time(x), e.moments[2] + time(x)^2)

    # Update node count
    e.node_count[x.node] += 1
    
    # Update word count
    e.mark_count .+= x.mark

    if recompute_posterior
        set_posterior!(model, k)
    end

    return k
end 

function event_list_summary(model::CablesModel)::Vector{CablesEventSummary}
    ev = events(model)
    return [
        CablesEventSummary((
            ind,
            ev[ind].sampled_position, 
            ev[ind].sampled_variance,
            ev[ind].sampled_amplitude,
            ev[ind].sampled_nodeproba,
            ev[ind].sampled_markproba

        )) 
        for ind in ev.indices
    ]
end
