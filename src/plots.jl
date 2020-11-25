_get_color(k::Int) = (k == -1) ? "black" : k

@recipe function f(data::Vector{Point})
    x = [position(pt)[1] for pt in data]
    y = [position(pt)[2] for pt in data]
    
    legend --> false
    seriestype --> :scatter
    x, y
end

@recipe function f(data::Vector{Point}, assignments::Vector{Int})
    colors = _get_color.(assignments)

    seriescolor --> colors
    data
end
