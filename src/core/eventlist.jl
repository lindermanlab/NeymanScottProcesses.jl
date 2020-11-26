"""
Dynamically re-sized array holding AbstractEvent structs.
events :
    Vector of SeqEvent structs, some may be empty.
indices :
    Sorted vector of integers, specifying the indices of
    non-empty SeqEvent structs. Does not contain duplicate
    integers. Note that `length(indices) <= length(events)`,
    with equality only holding if there are no empty events.
"""
EventList

function EventList(constructor, args)
    events = [constructor(args...)]  # one empty event
    indices = Int64[]
    return EventList(constructor, events, indices)
end

constructor_args(ev::EventList) = constructor_args(ev.events[1])

labels(ev::EventList) = ev.indices

Base.getindex(ev::EventList, i::Int64) = ev.events[i]

Base.length(ev::EventList) = length(ev.indices)

Base.iterate(ev::EventList) = (
    isempty(ev.indices) ? nothing : (ev.events[ev.indices[1]], 2)
)

Base.iterate(ev::EventList, i::Int64) = (
    (i > length(ev)) ? nothing : (ev.events[ev.indices[i]], i + 1)
)

"""
Finds either an empty SeqEvent struct, or creates a new one. Returns the index of the new
cluster.
"""
function add_event!(ev::EventList)

    # Check if any indices are skipped. If so, use the smallest skipped
    # integer as the index for the new event.
    i = 1
    for j in ev.indices

        # We have j == ev.indices[i].

        # If (indices[i] != i) then events[i] is empty.
        if i != j
            insert!(ev.indices, i, i)  # mark events[i] as occupied
            return i                   # i is the index for the new event.
        end

        # Increment to check if (i + 1) is empty.
        i += 1
    end

    # If we reached here without returning, then indices is a vector 
    # [1, 2, ..., K] without any skipped integers. So we'll use K + 1
    # as the new integer index.
    push!(ev.indices, length(ev.indices) + 1)

    # Create a new Event object if necessary.
    if length(ev.events) < length(ev.indices)
        args = constructor_args(ev)
        push!(ev.events, ev.constructor(args...))
    end

    # Return index of the empty SeqEvent struct.
    return ev.indices[end]
end


"""
Marks a SeqEvent struct as empty and resets its sufficient statistics. This does not delete 
the SeqEvent.
"""
function remove_event!(ev::EventList, index::Int64)
    reset!(ev.events[index])
    return deleteat!(ev.indices, searchsorted(ev.indices, index))
end


"""
Recompute sufficient statistics for all sequence events.
"""
function recompute!(
    model::NeymanScottModel,
    datapoints::Vector{<: AbstractDatapoint},
    assignments::AbstractVector{Int64}
)
    
    # Grab sequence event list.
    ev = events(model)

    # Reset all events to zero spikes.
    for k in ev.indices
        reset!(ev.events[k])
    end
    empty!(ev.indices)

    # Add spikes back to their previously assigned event.
    for (s, k) in zip(datapoints, assignments)
        
        # Skip background spikes.
        (k < 0) && continue

        # Check that event k exists.
        while k > length(ev.events)
            push!(ev.events, ev.constructor(constructor_args(ev)...))
        end

        # Add datapoint to event k.
        add_datapoint!(model, s, k, recompute_posterior=false)

        # Make sure that event k is marked as non-empty.
        j = searchsortedfirst(ev.indices, k)
        if (j > length(ev.indices)) || (ev.indices[j] != k)
            insert!(ev.indices, j, k)
        end
    end

    # Set the posterior, since we didn't do so when adding datapoints
    for k in ev.indices
        set_posterior!(model, k)
    end
end