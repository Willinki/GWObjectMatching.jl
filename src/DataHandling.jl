import FileIO: load
import ColorTypes: RGBA, RGB, Gray
import Images: N0f8
import MultivariateStats: MetricMDS, fit, predict, isotonic
using LinearAlgebra

Imagetype::Type = Union{Matrix{RGBA{N0f8}}, Matrix{RGB{N0f8}}, Matrix{Gray{N0f8}}}

"""
Main function:
- loads an image
- extracts meaningful pixels (as points in R^2)
- rescale points between 0 and 1
- applies random rotation
- randomly samples according to fraction or number n
Returns: 
- Vector{Vector{Float64, n=2}}
# TODO: set better path handling
"""
function load_image(filename::String ; n::Union{Int64, Float64}=1.)::Matrix{Float64}
    points = (
        load(filename) |> normalize_image |> get_coord_black_points
        |> rescale_convert_to_float |> randomRotate_points
    )
    n==1 && return points
    return undersample(points, n)
end

# UTILITY FUNCTIONS
"""
given a loaded image, returns a matrix of 0 and 1
"""
function normalize_image(img::Imagetype)::Matrix{Bool}
    """
    this is for png
    we return true if color = black or alpha!=0
    """
    function normalize_pixel(pix::RGBA{N0f8})::Bool
        return (pix.alpha ≈ 1) 
    end

    """
    this is for jpg and gifs
    we return true if have or color != (white)
    """
    function normalize_pixel(pix::RGB{N0f8})::Bool
        return (pix.r + pix.g + pix.b) ≈ 0
    end

    """
    this is for grayscale images
    we return true if have gray >0.5 (white)
    """
    function normalize_pixel(pix::Gray{N0f8})::Bool
        return pix.val > 0.5
    end

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
function rescale_convert_to_float(coords::Matrix{Int64})::Matrix{Float64}
    function rescale_vec(V::Vector{Int64})
        minV, maxV = minimum(V), maximum(V)
        minV == maxV && return V./minV
        return (V .- minV)./(maxV - minV)
    end
    return mapslices(rescale_vec, coords, dims=1)
end

"""
simple struct to define SO2 actions
"""
struct PlaneRotation
    action::Matrix{Float64}
    PlaneRotation(θ::T) where T<:Real = new([cos(θ) -sin(θ); sin(θ) cos(θ)])
end

"""
random rotation of points
"""
function randomRotate_points(coords::Matrix{Float64})
    R = PlaneRotation(rand(Float64)*2*π)
    return mapslices(x->R.action*x, coords, dims=2) 
end

"""
keeps only n-fraction of elements in v 
"""
function undersample(v::Matrix{Float64}, n::Float64)
    return undersample(v, round(Int64, n*size(v, 1)))
end

"""
keeps only n elements in v 
"""
function undersample(v::Matrix{Float64}, n::Int64)
    1 < n < size(v, 1) || throw(ArgumentError(
        "Provide a valid fraction or number for downsampling"
    ))
    return v[rand(1:size(v, 1), n), :]
end

"""
PCA to reconstruct images
"""
function reconstruct_points(distance_matrix::Matrix{Float64}; kwargs...)
    mds = fit(
        MetricMDS, distance_matrix;
        distances=true, maxoutdim=2, 
        kwargs...
    )
    return predict(mds)'
end

# PLOTTING FUNCTIONS
# TODO
# one for barycenter, one for interpolation
