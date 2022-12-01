import Distances: PreMetric, pairwise

"""
* C_{ij} is the distance/dissimilarity matrix of the graph, i.e. how much the i-th
and the j-th nodes are different

* μ_i is the information on how much the i-th node is important
"""
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
            μ=fill(1/size(C,1), size(C,1))::Vector{<:Real}
        ) 
        size(C, 1) != size(C, 2) && throw(ArgumentError(
            "The distance/dissimilarity matrix must be square."
        ))
        size(C, 1) != length(μ) && throw(ArgumentError(
            "The size of C and mu must coincide."
        ))
        any(μ .< 0) && throw(ArgumentError(
            "The entrance of mu must be non-negative."
        ))
        iszero(μ) && throw(ArgumentError(
            """mu can't be the zero vector, for the uniform distribution
            just use MMS with μ as default"""
        ))

        if typeof(μ) != Vector{Float64}
            μ = convert(Vector{Float64}, μ)
        end
        if typeof(C) != Matrix{Float64}
            C = convert(Matrix{Float64}, C)
        end

        if sum(μ) != 1
            μ .= (1/sum(μ)).*μ
            @info "We're changing mu in such a way it has sum 1."
        end

        return new(C,μ)
    end
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
        μ=fill(1/size(v), size(v))::Vector{<:Real}
    )
    isconcretetype(eltype(v)) && @warn "Vector dist is not homogeneous."
    return MetricMeasureSpace(distance_matrix(dist, v), μ)
end
