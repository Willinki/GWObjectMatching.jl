import Test: @test, @test_throws, @testset
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, NormalizedPositiveVector


@testset "MetricMeasureSpace" begin
    dim::Int64 = 10
    C          = rand(Float64, (2,2))
    μ          = rand(Float64, 2)
    M          = MetricMeasureSpace(C, DiscreteProbability(μ))

    # negative entries in mu raise error 
    @test_throws ArgumentError MetricMeasureSpace(C, DiscreteProbability(Float64[-1, 1]))

    # once constructed, mu sums to 1
    @test sum((M.μ).D)==1

    function mu_is_constant_when_default(C)
        mu_mms = MetricMeasureSpace(C).μ
        return all(mu_mms.D .== mu_mms.D[1])
    end
    @test mu_is_constant_when_default(C)

    # raise error when C and mu have different dimensions 
    @test_throws ArgumentError MetricMeasureSpace(C, DiscreteProbability(Float64[1, 1, 1]))

    # raise error when C is not square 
    @test_throws ArgumentError MetricMeasureSpace(Float64[[1] [1]])
    @test_throws ArgumentError MetricMeasureSpace(Float64[[1, 1, 1] [1, 1, 1]])

    # raise error if \mu is always zero
    @test_throws ArgumentError MetricMeasureSpace(C, DiscreteProbability(zeros(Float64, dim)))
end 
    
