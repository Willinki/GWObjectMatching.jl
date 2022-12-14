import Test: @test, @test_throws, @testset
import ObjectMatching: compute_C, loss, update_barycenters, GW_Cost, initialize_C, update_transport, compute_marginals
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, ConvexSum, NormalizedPositiveVector
using LinearAlgebra

@testset "Compute Barycenters" begin

    M = MetricMeasureSpace(rand(2,2))
    N = MetricMeasureSpace(rand(4,4))
    Cs_collection = [M,N]
    λs_collection = ConvexSum([0.4,0.6])
    Ts_collection = [rand(2,3),rand(4,3)]
    p = [0.2,0.4,0.4]
    Cp = initialize_C(p)
    Cp_updated = update_barycenters(Cp,Cs_collection,λs_collection)

    Q = MetricMeasureSpace(rand(10,10))
    Cq = MetricMeasureSpace(rand(20,20))

    function compute_C_returns_square_matrix(
            Cs_collection::Vector{MetricMeasureSpace},
            λs_collection::ConvexSum,
            Ts_collection::Vector{Matrix{Float64}},
            p::Vector{Float64},
            loss::loss
        )
        MMS = compute_C(λs_collection, Ts_collection, Cs_collection, p, loss)
        return size(MMS.C,1) == size(MMS.C,2)
    end

    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, loss("L2")
        ) 

    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, loss("KL")
        )

    @test_throws ArgumentError compute_C(
        λs_collection,Ts_collection,[MetricMeasureSpace([-1 1; 1 0]),N],p, loss("KL")
        )

    function p_is_strictly_positive(MMS::MetricMeasureSpace)
        mu_mms = MMS.μ
        return all(mu_mms.>0)
    end

    @test p_is_strictly_positive(
            update_barycenters(
                initialize_C([0.0, 1.0, 0.0, 2.0]),Cs_collection, λs_collection
            )
        )
    
    #initialize_C computes the trivial transport
    @test all((Cp.C).== Cp.C[1])

    #update_barycenters can be iterated
    @test typeof(
        update_barycenters(Cp_updated,Cs_collection,λs_collection)
    ) == MetricMeasureSpace

    #update_transport has the correct size
    function update_transport_has_the_correct_size(
        Cs::MetricMeasureSpace, Cp::MetricMeasureSpace, loss::loss, ϵ, tol
    )
        Ts = update_transport(Cs;Cp,loss,ϵ,tol).T
        return length(Cp.μ) == size(Ts,1) && length(Cs.μ) == size(Ts,2)
    end

    @test update_transport_has_the_correct_size(M, Cp, loss("L2"),1e-2,1e-8)

    function update_transport_approximates_original_marginals(
        Cs::MetricMeasureSpace, Cp::MetricMeasureSpace, loss::loss, ϵ, tol
    )
        updated_data_SK = update_transport(Cs;Cp,loss,ϵ,tol)
        Ts = updated_data_SK.T 
        p,q = compute_marginals(updated_data_SK)
        return (norm(p-Cp.μ,Inf) < 0.001) && (norm(q-Cs.μ,Inf) < 0.001)
    end

    @test update_transport_approximates_original_marginals(
        Q, Cq, loss("L2"),1e-2,1e-8
    )

end  
