import Test: @test, @test_throws, @testset
import ObjectMatching: compute_C
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, ConvexSum, NormalizedPositiveVector

@testset "barycenters" begin

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
        p::Vector{Float64}
        )
        C = compute_C(λs_collection,Ts_collection,Cs_collection,p)
        return size(C,1) == size(C,2)
    end

    @test compute_C_returns_square_matrix(Cs_collection, λs_collection, Ts_collection, p)

end  