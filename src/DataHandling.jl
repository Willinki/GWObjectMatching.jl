import FileIO: load
import ColorTypes: RGBA, RGB
import Images: N0f8
using LinearAlgebra

Imagetype::Type = Union{Matrix{RGBA{N0f8}}, Matrix{RGBA{N0f8}}}

"""
Main function:
- loads an image
- extracts meaningful pixels (as points in R^2)
- rescale points between 0 and 1
- applies random rotation
- randomly samples acco
Returns: 
- Vector{Vector{Float64, n=2}}
"""
function load_image(
    filename::String ;
    n::Union{Int64, Float64}=1.
    )::Vector{Vector{Float64}}
    # TODO: set better path handling
    points = (
        load(filename) |> normalize_image
        |> get_coord_black_points |> rescale |> randomRotate_points
    )
    n==1 &&
        return points
    return undersample(points, n)
end

# UTILITY FUNCTIONS
"""
given a loaded image, returns a matrix of 0 and 1
"""
function normalize_image(img::Imagetype)::Matrix{Bool}
    """
    this is for png
    we return true if color != (white) or alpha!=0
    """
    function normalize_pixel(pix::RGBA{N0f8})::Bool
        return !(pix.alpha == 0 || ((pix.r + pix.g + pix.b) ≈ 3))
    end

    """
    this is for jpg and gifs
    we return true if have or color != (white)
    """
    function normalize_pixel(pix::RGB{N0f8})::Bool
        return !((pix.r + pix.g + pix.b) ≈ 3)
    end
    # applies and returns 
    return map(normalize_pixel, img)::Matrix{Bool}
end

"""
given a matrix bools, returns the coordinates
return type is matrix with 2 columns and N rows
"""
function get_coord_black_points(img::Matrix{Bool})::Matrix{Int64}
    return reduce(vcat, [[x[1] x[2]] for x in findall(img)])
end

"""
each coordinate is normalized to be between 0 and 1
"""
function rescale(coords::Matrix{Int64})::Matrix{Float64}
    function rescale_vec(V::Vector{Int64})
        minV, maxV = minimum(V), maximum(V)
        minV == maxV && return V./minV
        return (V .- minV)/(maxV - minV)
    end
    return mapslices(rescale_vec, coords, dims=1)
end

"""
random rotation of points
"""
function randomRotate_points(coords::Matrix{Float64})
    R = PlaneRotation(rand(Float64)*2*π)
    return mapslices(x->R.action*x, coords, dims=2) 
end

"""
simple struct to define SO2 actions
"""
struct PlaneRotation
    action::Matrix{Float64}
    PlaneRotation(θ::T) where T<:Real = new([cos(θ) -sin(θ); sin(θ) cos(θ)])
end

"""
keeps only n-fraction of elements in v 
"""
function undersample(v::Vector{Vector{Float64}}, n::Float64)
    throw("not implemented")
end

"""
keeps only n elements in v 
"""
function undersample(v::Vector{Vector{Float64}}, n::Int64)
    throw("not implemented")
end

"""
PCA to reconstruct images
"""
function distances_to_points(dist_matrix::Matrix{Float64})
    throw("Not implemented")
end

# PLOTTING FUNCTIONS
# TODO
