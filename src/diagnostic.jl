function metric(ω1, ω2)
    return ω1 == ω2
end

function cooccupancy_matrix(assignments::Vector{Int})
    return metric.(assignments, assignments')
end

function cooccupancy_matrix(assignments_history::Vector{Vector{Int}})
    n = length(first(assignments_history))

    C = zeros(n, n)
    for assignments in assignments_history
        C .+= cooccupancy_matrix(assignments)
    end

    return C / length(assignments_history)
end