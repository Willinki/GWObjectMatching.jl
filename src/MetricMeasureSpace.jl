import Distances: PreMetric, pairwise

"""
* C_{ij} is the distance/dissimilarity matrix of the graph, i.e. how much the i-th
and the j-th nodes are different

* μ_i is the information on how much the i-th node is important
"""
abstract type NormalizedPositiveVector end

struct DiscreteProbability <: NormalizedPositiveVector
    D::Vector{Float64}
    """
    inner constructor. It forces the vector to be non-negative and
    to have sum 1.
    """
    function DiscreteProbability(D::Vector{Float64})

        any(D .< 0) && throw(ArgumentError(
            "The entrance of D must be non-negative."
        ))
        iszero(D) && throw(ArgumentError(
            """D can't be the zero vector."""
        ))

        if typeof(D) != Vector{Float64}
            D = convert(Vector{Float64}, D)
        end

        if sum(D) != 1
            D .= (1/sum(D)).*D
            @info "We're changing D in such a way it has sum 1."
        end

        return new(D)  
    end
end

"""
inner constructor. It forces the vector to be non-negative and
to have sum 1.
"""
struct ConvexSum <: NormalizedPositiveVector
    v::Vector{Float64}
    function ConvexSum(D::Vector{Float64})
        any(D .< 0) && throw(ArgumentError("The entrance of D must be non-negative."))
        iszero(D) && throw(ArgumentError("""D can't be the zero vector."""))
        if typeof(D) != Vector{Float64}
            D = convert(Vector{Float64}, D)
        end
        if sum(D) ≈ 1
            D .= (1/sum(D)).*D
            @info "We're changing D in such a way it has sum 1."
        end
        return new(D)
    end
end

struct MetricMeasureSpace
    C::Matrix{Float64}
    μ::Vector{Float64}

    """
    inner constructor. It forces the following behavior
    - typeof C must be Matrix{Float64}
    - mu has the uniform distribution as standard value
    - The entrance of mu must be non-negative
    - C must be a square matrix and the size of C and mu must coincide
    - mu cannot be 
    - We force mu to sum at 1, a warning is raised in that case
    """
    function MetricMeasureSpace(
            C::Matrix{<:Real},
            μ=fill(1/size(C,1), size(C,1))::Vector{Float64}
        ) 
        size(C, 1) != size(C, 2) && throw(ArgumentError(
            "The distance/dissimilarity matrix must be square."
        ))
        size(C, 1) != length(μ) && throw(ArgumentError(
            "The size of C and mu must coincide."
        ))

        prob = DiscreteProbability(μ)
        if typeof(C) != Matrix{Float64}
            C = convert(Matrix{Float64}, C)
        end
        return new(C,prob.D)
    end #innerconstructor
end #struct

distance_matrix(dist::Function, v::Vector) = [dist(x,y) for x in v, y in v] 
distance_matrix(dist::PreMetric, v::Vector) = pairwise(dist, v) 
"""
External constructor. Given an array and dissimilarity function between its
elements calculates the distance matrix and constructs MetricMeasureSpace.
We chose isconcretetype to allow for strings arrays. 
"""
function MetricMeasureSpace(
        dist::Union{Function, PreMetric},
        v::Vector,
        μ=fill(1/size(v), size(v))::Vector{Float64}
    )
    isconcretetype(eltype(v)) && @warn "Vector dist is not homogeneous."
    return MetricMeasureSpace(distance_matrix(dist, v), DiscreteProbability(μ).D)
end

distance_matrix(dist::Function, v::Matrix{Float64}) = [dist(x,y) for x in eachrow(v), y in eachrow(v)] 
distance_matrix(dist::PreMetric, v::Matrix{Float64}) = pairwise(dist, eachrow(v)) 
"""
External constructor. Given a matrix of points and dissimilarity function between its
elements calculates the distance matrix and constructs MetricMeasureSpace.
We chose isconcretetype to allow for strings arrays. 
"""
function MetricMeasureSpace(
        dist::Union{Function, PreMetric},
        v::Matrix{Float64},
        μ=fill(1/size(v), size(v)::Vector{Float64}
        )
    )
    return MetricMeasureSpace(distance_matrix(dist, v), DiscreteProbability(μ).D)
end
