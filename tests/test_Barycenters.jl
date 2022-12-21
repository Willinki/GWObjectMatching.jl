import Test: @test, @test_throws, @testset
import ObjectMatching: compute_C, Loss, update_barycenters, GW_Cost, init_C, init_Ts
import ObjectMatching: update_transport, compute_marginals, GW_barycenters
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, ConvexSum, NormalizedPositiveVector
using LinearAlgebra

@testset "Compute Barycenters" begin
    ## parameters
    ϵ = 0.008
    Cp_niter = 1 
    Ts_tol = 1e-4
    SK_tol = 1e-4
    ## simulating inputs
    M = MetricMeasureSpace(rand(20,20))
    N = MetricMeasureSpace(rand(40,40))
    O = MetricMeasureSpace(rand(32,32))
    Cs_collection = [M,N,O]
    λs_collection = ConvexSum([0.4,0.5,0.1])
    Ts_collection = [rand(20,30),rand(40,30),rand(32,30)]
    lossL2 = Loss("L2")
    lossKL = Loss("KL")
    ## FIXTURES - we defined two for convergence problems
    q = rand(30)
    q2 = rand(100)
    p = (1/sum(q))*q
    p2 = (1/sum(q2))*q2
    Cp = init_C(p)
    Cp_updated = update_barycenters(
        Cp ; 
        Cs_collection = Cs_collection,
        λs_collection = λs_collection,
        loss = lossL2,
        ϵ = ϵ,
        Ts_tol = Ts_tol,
        SK_tol = SK_tol
    )

    function compute_C_returns_square_matrix(
            Cs_collection::Vector{MetricMeasureSpace},
            λs_collection::ConvexSum,
            Ts_collection::Vector{Matrix{Float64}},
            p::Vector{Float64},
            loss::Loss
        )
        MMS = compute_C(λs_collection, Ts_collection, Cs_collection, p, loss)
        return size(MMS.C,1) == size(MMS.C,2)
    end
    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, lossL2
    ) 
    @test compute_C_returns_square_matrix(
        Cs_collection, λs_collection, Ts_collection, p, lossL2
    )
    @test_throws ArgumentError compute_C(
        λs_collection,Ts_collection,[MetricMeasureSpace([-1 1; 1 0]),N], p, lossKL
    )

    function p_is_strictly_positive(MMS::MetricMeasureSpace)
        mu_mms = MMS.μ
        return all(mu_mms.>0)
    end
    @test p_is_strictly_positive(
        update_barycenters(
            init_C([0.0, 1.0, 0.0, 2.0]);
            Cs_collection = Cs_collection, 
            λs_collection = λs_collection,
            loss = lossL2,
            ϵ = ϵ, Ts_tol = Ts_tol, SK_tol= SK_tol
        )
    )
    
    ##initialize_C computes the trivial transport
    #@test all((Cp.C).== Cp.C[1])

    #update_barycenters can be iterated
    @test MetricMeasureSpace == typeof(
        update_barycenters(
            init_C([0.0, 1.0, 0.0, 2.0]);
            Cs_collection = Cs_collection, 
            λs_collection = λs_collection,
            loss = lossL2,
            ϵ = ϵ, Ts_tol = Ts_tol, SK_tol= SK_tol
       )
    ) 
    @test MetricMeasureSpace == typeof(
        update_barycenters(
            init_C([0.0, 1.0, 0.0, 2.0]);
            Cs_collection = Cs_collection, 
            λs_collection = λs_collection,
            loss = lossL2,
            ϵ = ϵ, Ts_tol = Ts_tol, SK_tol= SK_tol
        )
    ) 

    function update_transport_has_the_correct_size(
            Cs::MetricMeasureSpace,
            Cp::MetricMeasureSpace,
            loss::Loss,
            ϵ::Float64, SK_tol::Float64
        )
        Ts = update_transport(
            init_Ts(Cp, Cs); 
            Cs=Cs, Cp=Cp, loss=loss, ϵ=ϵ, SK_tol=SK_tol
        )
        return length(Cp.μ) == size(Ts,1) && length(Cs.μ) == size(Ts,2)
    end
    @test update_transport_has_the_correct_size(M, Cp, lossL2, 1e-2, 1e-8)

    function update_transport_approximates_original_marginals(
            Cs::MetricMeasureSpace,
            Cp::MetricMeasureSpace,
            loss::Loss,
            ϵ::Float64, 
            SK_tol::Float64
        )
        Ts = init_Ts(Cp,  Cs)
        Ts = update_transport(
            Ts; 
            Cs=Cs, Cp=Cp, loss=loss, ϵ=ϵ, SK_tol=SK_tol
        )
        p1,q1 = compute_marginals(Ts)
        return (norm(p1-Cp.μ, Inf) < SK_tol) && (norm(q1-Cs.μ, Inf) < SK_tol)
    end
    @test update_transport_approximates_original_marginals(M, N, lossL2, ϵ, SK_tol)

    function GW_barycenters_has_the_correct_size(
            n::Int64, 
            Cs_collection::Vector{MetricMeasureSpace}, 
            λs_collection::ConvexSum,
            p::Vector{Float64} = fill(1/n,n),
            loss::Loss = loss("L2")
        )
        Cp = GW_barycenters(
            Cs_collection,
            λs_collection;
            n_points = n,
            p = p,
            loss = Loss("L2"),
            ϵ = ϵ, Cp_niter = Cp_niter, Ts_tol = Ts_tol, SK_tol = SK_tol,
        )
        return (size(Cp.C,1) == size(Cp.C,2) && size(Cp.C,1) == n)
    end
    @test typeof(GW_barycenters(
            Cs_collection,
            λs_collection;
            n_points=100,
            p = fill(1/100, 100),
            loss = Loss("L2"),
            ϵ = ϵ, Cp_niter = Cp_niter, Ts_tol = Ts_tol, SK_tol = SK_tol,
    )) == MetricMeasureSpace
    @test GW_barycenters_has_the_correct_size(100, Cs_collection, λs_collection)

    function GW_baryc_has_the_correct_prob_measure(
            n::Int64, 
            Cs_collection::Vector{MetricMeasureSpace}, 
            λs_collection::ConvexSum,
            p=fill(1/n,n)::Vector{Float64},
            loss=Loss("L2")::loss,
            ϵ = ϵ, Cp_niter = Cp_niter, Ts_tol = Ts_tol, SK_tol = SK_tol,
        )
        Cp = GW_barycenters(n,Cs_collection,λs_collection,p,loss,ϵ,Cp_niter,Ts_tol,SK_tol)
        return (Cp.μ == DiscreteProbability(p).D)
    end
    @test GW_baryc_has_the_correct_prob_measure(100,Cs_collection,λs_collection,p2)
end  
