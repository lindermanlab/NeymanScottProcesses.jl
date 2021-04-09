"""
    Region

Abstract type representing some subset of a (potentially non-Euclidean) space.
"""
abstract type Region end

"""
    sample(r::Region)

Uniformly samples an element of `r`.
"""
sample(r::Region) = notimplemented()

"""
    volume(r::Region)

Computes the volume of the region `r`, typically this would
be defined with respect to the Lebesgue measure.
"""
volume(r::Region) = notimplemented()

# ===
# Triangular Regions in 2D space
# ===

"""
    Triangle{R <: Real} <: Region

Represents
"""
struct Triangle{R <: Real}
    # The three points defining the triangle are:
    #   {(x1, y1), (x2, y2), (x3, y3)}
    # The coordinates stored in this order:
    #    x1, y1, x2, y2, x3, y3
    coordinates::Tuple{R,R,R,R,R,R}
end

function Triangle{R}(
        a::Tuple{R,R},
        b::Tuple{R,R},
        c::Tuple{R,R}
    ) where {R <: Real}
    return Triangle((a..., b..., c...))
end

function Base.in(pt::Vector, tri::Triangle)

    # https://mathworld.wolfram.com/TriangleInterior.html

    # Check input.
    if len(pt) != 2
        throw(InputError("Triangular regions are only valid for 2d observations"))
    end
    x, y = pt

    # Unpack 2d coordinates for 3 endpoints of the triangle.
    x1, y1, x2, y2, x3, y3 = tri.coordinates

    # Compute Barycentric coordinates of (x, y).
    alpha = (
        ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) /
        ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    )
    beta = (
        ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) /
        ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    )
    gamma = 1.0 - alpha - beta

    # The point is inside the triangle if all barcentric coords are positive.
    return (alpha > 0) && (beta > 0) && (gamma > 0)

end

function volume(tri::Triangle)
    x1, y1, x2, y2, x3, y3 = tri.coordinates
    return 0.5 * abs(
        x1 * (y2 - y3) + x2 * (y3 - y2) + x3 * (y1 - y2)
    )
end

function sample(tri::Triangle)
    x1, y1, x2, y2, x3, y3 = tri.coordinates

    # Reference:
    #   Osada R, Funkhouser T, Chazelle B, Dobkin D.
    #   (2002). "Shape Distributions." ACM Transactions
    #   on Graphics, Vol. 21, No. 4. See section 4.2
    r1, r2 = rand(), rand()
    sqrt_r1 = sqrt(r1)
    s1 = 1 - sqrt_r1
    s2 = sqrt_r1 * (1 - r2)
    s3 = sqrt_r1 * r2

    p = zeros(2)
    p[1] = s1 * x1 + s2 * x2 + s3 * x3
    p[2] = s1 * y1 + s2 * y2 + s3 * y3

    return p
end


# ===
# Box Regions
# ===

"""
    Box <: Region

Represents an N-dimensional box.
"""
struct Box <: Region
    bounds::Vector{Float64}
    _distr::Distribution
end

Box(bounds::Vector) = Box(
    bounds, Product([Uniform(0, x) for x in bounds])
)

Base.ndims(b::Box) = length(b.bounds)

volume(b::Box) = prod(b.bounds)

function Base.in(vector::Vector{Float64}, box::Box)
    for (x, b) in zip(vector, box.bounds)
        if (x > b)
            return false
        end
    end
    return true
end

sample(b::Box) = rand(b._distr)

sample(b::Box, n::Integer) = [rand(b._distr) for i in 1:n]

observations_type(b::Box) = Vector{Float64}

# ===
# Spherical Regions
# ===

"""
    Sphere{N <: Integer} <: Region

Represents an N-dimensional sphere.
"""
struct Sphere <: Region
    center::Vector{Float64}
    radius::Float64
end
# Sphere(center::AbstractVector, radius::Float64) = (
#     Sphere(SVector(length(center), center), radius)
# )
function volume(s::Sphere)
    n = ndims(s)
    return s.radius^n * π^(0.5 * n) / gamma(1 + 0.5 * n)
end
Base.ndims(s::Sphere) = length(s.center)
Base.in(pt::Vector{Float64}, s::Sphere) = (
    norm(s.center .- pt) <= s.radius
)

function sample(s::Sphere)
    u = randn(ndims(s))
    return s.center + u * (rand() * s.radius / norm(u))
end

observations_type(s::Sphere) = Vector{Float64}

function plot!(ax::PyObject, s::Sphere; kwargs...)
    @assert length(s.center) == 2
    ax.add_patch(
        plt.Circle(s.center, s.radius; kwargs...)
    )
end

# ===
# Spike train regions
# ===


"""
I think this is how one would proceed for PPSeq?
"""
struct IntegerMarkedInterval{I <: Integer, W <: AbstractInterval} <: Region
    index::I
    interval::W
end
function Base.in(
    x::Tuple{F, I},
    int::IntegerMarkedInterval{I, W}
) where {I <: Integer, F <: Real, W <: AbstractInterval}
    # Check x is included in interval and has the correct integer mark.
    return (x[1] in int.interval) && (x[2] == int.index)
end
volume(int::IntegerMarkedInterval) = width(int.interval)
sample(int::IntegerMarkedInterval) = (
    rand() * (int.interval.right - int.interval.left) + int.interval.left
)

# ===
# Collections / Unions of non-overlapping regions.
# ===


"""
    RegionCollection(regions) <: Region

Represents a union of non-overlapping region structs
(specified by a vector `regions`). Relaxing the
assumption that the regions do not overlap would be
non-trivial except in special cases.
"""
struct RegionCollection <: Region
    regions::Vector{<: Region}
end

# TODO: add constructor to test that inner and outer region
#       have the observations_type

volume(rc::RegionCollection) = (
    sum(volume(r) for r in rc.regions)
)

Base.in(x, rc::RegionCollection) = (
    any(r -> (x in r), rc.regions)
)

Base.length(rc::RegionCollection) = length(rc.regions)

Base.iterate(rc::RegionCollection) = iterate(rc.regions)

Base.iterate(rc::RegionCollection, state) = iterate(rc.regions, state)

function sample(rc::RegionCollection)

    # Randomly pick which region to sample in proportion to volume.
    i = sample(pweights([volume(region) for region in rc]))

    # Draw a sample from the selected region
    return sample(rc.region[i])

end

function plot!(ax::PyObject, rc::RegionCollection; kwargs...)
    for region in rc
        plot!(ax, region; kwargs...)
    end
end

# ===
# Complements
# ===

"""
    ComplementRegion(masked_region, domain) <: Region

Represents the complement of a small region (`masked_region`), 
which is completely contained in a larger region
(`domain`).
"""
struct ComplementRegion <: Region
    masked_region::Region
    domain::Region
end

# TODO: add constructor to test that masked_region has the
#       same observations_type as domain. Also check that
#       masked_region is contained inside domain.

volume(c::ComplementRegion) = (
    volume(c.domain) - volume(c.masked_region)
)

Base.in(x, c::ComplementRegion) = (
    (x in c.domain) && !(x in c.masked_region)
)

function sample(c::ComplementRegion)
    # Do rejection sampling.
    x = sample(c.domain)
    while (x in c.masked_region)
        x = sample(c.domain)
    end
    return x
end

observations_type(c::ComplementRegion) = observations_type(c.domain)


# """
#     data_inside, data_outside = split_data(data, region)

# Split vector of datapoints (`data`) into two non-overlapping subsets.
# Datapoints inside `region` are placed in `data_inside`. Datapoints
# outside `region` are placed in `data_outside`.
# """
# function split_data(
#     data::Vector{D},
#     region::Region
# ) where {D}

#     data_inside = D[]
#     data_outside = D[]

#     for x in data
#         if x in region
#             push!(data_inside, x)
#         else
#             push!(data_outside, x)
#         end
#     end

#     return data_inside, data_outside
# end




"""
    sample_random_spheres(b::Box, n_spheres, fraction_filled)

Samples randomly positioned and non-overlapping spheres
within N-dimensional box `b`. Returns a `RegionCollection`
struct containing `n_spheres` regions. The total volume of
the spheres is given by `volume(b) * fraction_filled`.
"""
function sample_random_spheres(
        b::Box,
        n_spheres::Integer,
        fraction_filled::Real;
        max_rejects::Integer=1_000_000
    )


    # Compute radii of the spheres.
    #
    # volume(b) * fraction_filled = 
    #       n_spheres * radii^N * π^(0.5 * N) / gamma(1 + 0.5 * N)
    N = ndims(b)
    radii = (
        (volume(b) * fraction_filled * gamma(1 + 0.5 * N)) / (n_spheres * π^(0.5 * N))
    ) ^ (1 / N)
    diam = 2 * radii
            

    @assert minimum(b.bounds) > radii

    spheres = Sphere[]
    num_rejects = 0

    while (length(spheres) < n_spheres) && (num_rejects <= max_rejects)
        # Sample center
        center = (rand(N) .* (b.bounds .- diam)) .+ radii
        
        # Check if sphere overlaps with any existing spheres
        overlaps = any(s -> 0.5 * norm(center - s.center) < radii, spheres)

        # Add sphere to list if it doesn't overlap.
        if overlaps
            num_rejects += 1
        else
            push!(spheres, Sphere(center, radii))
        end
    end

    if num_rejects > max_rejects
        @warn "Sampling spheres did not converge"
    end

    return RegionCollection(spheres)
end
