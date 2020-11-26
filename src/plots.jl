_get_color(k::Int) = (k == -1) ? "black" : k

@recipe function f(data::Vector{RealObservation{N}}) where {N}
    @assert N === 2
    x = [position(pt)[1] for pt in data]
    y = [position(pt)[2] for pt in data]
    
    legend --> false
    seriestype --> :scatter
    x, y
end

@recipe function f(data::Vector{RealObservation{N}}, assignments::Vector{Int}) where {N}
    colors = _get_color.(assignments)

    seriescolor --> colors
    data
end
