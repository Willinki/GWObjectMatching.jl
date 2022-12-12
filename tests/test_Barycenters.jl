import Test: @test, @test_throws, @testset
import ObjectMatching: compute_C, loss, update_barycenters, GW_Cost
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, ConvexSum, NormalizedPositiveVector

@testset "Compute Barycenters" begin

    M = MetricMeasureSpace(rand(2,2))
    N = MetricMeasureSpace(rand(3,3))
    Cs_collection = [M,N]
    λs_collection = ConvexSum([0.4,0.6])
    Ts_collection = [rand(2,3),rand(3,3)]
    p = [0.2,0.4,0.4]

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
        return mu_mms[1]>0
    end

    @test p_is_strictly_positive(
        update_barycenters(Cs_collection, λs_collection, [0.0, 1.0, 0.0, 2.0])
        )

end  
