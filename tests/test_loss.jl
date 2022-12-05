import Test: @test, @test_throws, @testset
import ObjectMatching: loss, GW_Cost
import ObjectMatching: MetricMeasureSpace 


@testset "loss" begin

    L = loss("L2")
    M = MetricMeasureSpace(rand(2,2))

    @test_throws ArgumentError loss("any string")

    @test_throws ArgumentError GW_Cost(L, M, MetricMeasureSpace(rand(2,2)), rand(3,2))

    @test_throws ArgumentError GW_Cost(L, M, MetricMeasureSpace(rand(3,3)), rand(3,2))

    @test_throws ArgumentError GW_Cost(L, M, MetricMeasureSpace(rand(2,2)), rand(3,3))

    @test_throws ArgumentError GW_Cost(L, M, MetricMeasureSpace(rand(3,3)), rand(3,2))

    @test typeof(GW_Cost(loss("L2"), MetricMeasureSpace(rand(2,2)), MetricMeasureSpace(rand(3,3)), rand(2,3))) == Matrix{Float64}

end  