@recipe function f(masks::Vector{CircleMask{N}}; d1=1, d2=2, detail=100) where {N}
    num_masks = length(masks)
    x, y = zeros(detail, num_masks), zeros(detail, num_masks)

    for (i, msk) in enumerate(masks)
        x_mask, y_mask = _circle_shape((msk.center[d1], msk.center[d2]), msk.radius, detail)
        x[:, i] = x_mask
        y[:, i] = y_mask
    end

    seriescolor --> "red"
    label --> nothing
    fillalpha --> 0.2
    seriestype --> :shape
    x, y
end

@recipe function f(data::Vector{RealObservation{N}}; d1=1, d2=2) where {N}
    x = [position(pt)[d1] for pt in data]
    y = [position(pt)[d2] for pt in data]
    
    label --> nothing
    seriestype --> :scatter
    x, y
end


@recipe function f(data::Vector{RealObservation{N}}, assignments::Vector{Int}) where {N}
    colors = _get_color.(assignments)

    seriescolor --> colors
    data
end

_get_color(k::Int) = (k == -1) ? "black" : k


function _circle_shape(x, r, n=100)
    θ = LinRange(0, 2π, n)
    return x[1] .+ r*cos.(θ), x[2] .+ r*sin.(θ)
end
