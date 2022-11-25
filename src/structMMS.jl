struct MetricMeasureSpace
    C::Matrix{Float64}
    mu::Vector{Float64}
        """
        C_{ij} is the distance/dissimilarity matrix of the graph, i.e. how much the i-th and the j-th nodes are different
        mu_i is the information on how much the i-th node is important
        """

    ####inner constructor, mu has the uniform distribution as standard value####
    #function MMS(C::Matrix{Float64},mu::Vector{Float64}) 
    function MetricMeasureSpace(C::Matrix{Float64},mu=(1/size(C)[1])*ones(size(C)[1])::Vector{Float64}) 

        #C must be a square matrix and the size of C and mu must coincide
        if size(C)[1] != size(C)[2]
            error("The distance/dissimilarity matrix must be square.")
        elseif size(C)[1]!= length(mu)
            throw(ArgumentError("The size of C and mu must coincide."))
        end

        ####The entrance of mu must be non-negative####
        for i = 1:length(mu)
            if mu[i]<0 
                throw(ArgumentError("The entrance of mu must be non-negative."))
            end
        end

        ####At least an entrance of mu must be positive####
        if iszero(mu)
            throw(ArgumentError("mu can't be the zero vector, for the uniform distribution just use MMS with only C as input."))
        end

        ####We force mu to sum at 1, and inform the user that we're changing mu if this is not the case####
        if sum(mu) != 1
            mu = (1/sum(mu))*mu
            @info "We're changing mu in such a way it has sum 1."
        end

        ####typeof C must be Matrix{Float64}####
        if typeof(C) == Matrix{Int64}                      #in realtà mi sembra che julia già lo faccia da solo 
            #convert(Matrix{Float64},C)
        end

        ####typeof mu must be Vector{Float64}####
        if typeof(mu) == Vector{Int64}                     #in realtà mi sembra che julia già lo faccia da solo 
        #    convert(Vector{Float64},mu)
        end
        return new(C,mu)
    end

end

