struct MetricMeasureSpace
    """
    * C_{ij} is the distance/dissimilarity matrix of the graph, i.e. how much the i-th
    and the j-th nodes are different

    * mu_i is the information on how much the i-th node is important
    """
    C::Matrix{Float64}
    mu::Vector{Float64}

    function MetricMeasureSpace(
        C::Matrix{<:Real},
        mu=(1/size(C)[1])*ones(size(C)[1])::Vector{<:Real}
        ) 
        """
        inner constructor. It forces the following behavior
        - typeof C must be Matrix{Float64}
        - mu has the uniform distribution as standard value
        - The entrance of mu must be non-negative
        - C must be a square matrix and the size of C and mu must coincide
        - At least an entrance of mu must be positive
        - We force mu to sum at 1, a warning is raised in that case
        """
        if size(C)[1] != size(C)[2]
            error("The distance/dissimilarity matrix must be square.")
        elseif size(C)[1]!= length(mu)
            throw(ArgumentError("The size of C and mu must coincide."))
        end

        any(x->x<0, mu) && throw(ArgumentError("The entrance of mu must be non-negative."))

        if iszero(mu)
            throw(ArgumentError(
                """mu can't be the zero vector, for the uniform distribution
                just use MMS with only C as input."""
            ))
        end

        if sum(mu) != 1
            mu = (1/sum(mu))*mu
            @info "We're changing mu in such a way it has sum 1."
        end

        if typeof(C) == Matrix{Int64}                      #in realtà mi sembra che julia già lo faccia da solo 
            #convert(Matrix{Float64},C)
        end

        ####typeof mu must be Vector{Float64}####
        if typeof(mu) == Vector{Int64}                     #in realtà mi sembra che julia già lo faccia da solo 
        #    convert(Vector{Float64},mu)
        end
        return new(C,mu)
    end
end #struct
