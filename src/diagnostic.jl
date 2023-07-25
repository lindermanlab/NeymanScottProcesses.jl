function metric(ω1, ω2)
    return ω1 == ω2
end

function cooccupancy_matrix(assignments::Vector{Int})
    return metric.(assignments, assignments')
end

function cooccupancy_matrix(assignments_history)
    n = length(first(assignments_history))

    C = zeros(n, n)
    for assignments in assignments_history
        C .+= cooccupancy_matrix(assignments)
    end

    return C / length(assignments_history)
end

function get_runtime(history)
    return (history.time .- first(history.time))
end