import Test: @test, @test_throws, @testset
import ObjectMatching: compute_C, loss, update_barycenters, GW_Cost, initialize_C
import ObjectMatching: update_transport, compute_marginals, GW_barycenters
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, ConvexSum, NormalizedPositiveVector
using LinearAlgebra

@testset "Compute Barycenters" begin
    M = MetricMeasureSpace(rand(20,20))
    N = MetricMeasureSpace(rand(40,40))
    O = MetricMeasureSpace(rand(32,32))
    Cs_collection = [M,N,O]
    λs_collection = ConvexSum([0.4,0.5,0.1])
    Ts_collection = [rand(20,30),rand(40,30),rand(32,30)]
    lossL2 = loss("L2")
    lossKL = loss("KL")
    ϵ = 1e-2
    tol = 1e-8
    q = rand(30)
    p = (1/sum(q))*q
    Cp = initialize_C(p)
    Cp_updated = update_barycenters(
        Cp;Cs_collection,λs_collection,loss=lossL2,ϵ,tol
    )
    Q = MetricMeasureSpace(rand(10,10))
    Cq = MetricMeasureSpace(rand(20,20))

    function compute_C_returns_square_matrix(
            Cs_collection::Vector{MetricMeasureSpace},
            λs_collection::ConvexSum,
            Ts_collection::Vector{Matrix{Float64}},
            p::Vector{Float64},
            loss::loss
        )
        MMS = compute_C(λs_collection, Ts_collection, Cs_collection, p, lossL2)
        return size(MMS.C,1) == size(MMS.C,2)
    end
    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, lossL2
        ) 
    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, lossL2
        )
    @test_throws ArgumentError compute_C(
        λs_collection,Ts_collection,[MetricMeasureSpace([-1 1; 1 0]),N],p, lossKL
        )

    function p_is_strictly_positive(MMS::MetricMeasureSpace)
        mu_mms = MMS.μ
        return all(mu_mms.>0)
    end
    @test p_is_strictly_positive(
            update_barycenters(
                initialize_C([0.0, 1.0, 0.0, 2.0]);
                Cs_collection, λs_collection,loss=lossL2,ϵ,tol
            )
        )
    
    #initialize_C computes the trivial transport
    @test all((Cp.C).== Cp.C[1])

    #update_barycenters can be iterated
    @test MetricMeasureSpace == typeof(
        update_barycenters(Cp_updated;Cs_collection,λs_collection,loss=lossL2,ϵ,tol)
    ) 
    @test MetricMeasureSpace == typeof(
        update_barycenters(Cp_updated;Cs_collection,λs_collection,loss=lossKL,ϵ,tol)
    ) 

    function update_transport_has_the_correct_size(
            Cs::MetricMeasureSpace,
            Cp::MetricMeasureSpace,
            loss::loss,
            ϵ, tol
        )
        Ts = update_transport(Cs;Cp,loss,ϵ,tol).T
        return length(Cp.μ) == size(Ts,1) && length(Cs.μ) == size(Ts,2)
    end
    @test update_transport_has_the_correct_size(M, Cp, lossL2,1e-2,1e-8)

    function update_transport_approximates_original_marginals(
            Cs::MetricMeasureSpace,
            Cp::MetricMeasureSpace,
            loss::loss,
            ϵ, tol
        )
        updated_data_SK = update_transport(Cs;Cp,loss,ϵ,tol)
        Ts = updated_data_SK.T 
        p1,q1 = compute_marginals(updated_data_SK)
        return (norm(p1-Cp.μ, Inf) < 1e-8) && (norm(q1-Cs.μ, Inf) < 1e-8)
    end
    @test update_transport_approximates_original_marginals(
        Q, Cq, lossL2, 1e-2, 1e-8
    )

    function GW_barycenters_has_the_correct_size(
        n::Int64, 
        Cs_collection::Vector{MetricMeasureSpace}, 
        λs_collection::ConvexSum,
        p=fill(1/n,n)::Vector{Float64},
        loss=loss("L2")::loss,
        ϵ=1e-2::Float64,
        tol=1e-8::Float64,
        niter=20::Int64
    )
        Cp = GW_barycenters(n,Cs_collection,λs_collection,p,loss,ϵ,tol,niter).C
        return (size(Cp,1) == size(Cp,2) && size(Cp,1) == n)
    end
    @test typeof(GW_barycenters(30, Cs_collection, λs_collection)) == MetricMeasureSpace
    @test GW_barycenters_has_the_correct_size(30, Cs_collection, λs_collection)
    @test GW_barycenters_has_the_correct_size(30, Cs_collection, λs_collection, p, lossKL)
    @test_throws ArgumentError GW_barycenters(50, Cs_collection, λs_collection, p)

    function GW_baryc_has_the_correct_prob_measure(
        n::Int64, 
        Cs_collection::Vector{MetricMeasureSpace}, 
        λs_collection::ConvexSum,
        p=fill(1/n,n)::Vector{Float64},
        loss=loss("L2")::loss,
        ϵ=1e-2::Float64,
        tol=1e-8::Float64,
        niter=20::Int64
    )
        Cp = GW_barycenters(n,Cs_collection,λs_collection,p,loss,ϵ,tol,niter)
        return (Cp.μ == DiscreteProbability(p).D)
    end
    @test GW_baryc_has_the_correct_prob_measure(30,Cs_collection,λs_collection,rand(30))
    @test GW_baryc_has_the_correct_prob_measure(30,Cs_collection,λs_collection,p,lossKL)
    @test typeof(GW_barycenters(1000, Cs_collection, λs_collection))==MetricMeasureSpace
end  
