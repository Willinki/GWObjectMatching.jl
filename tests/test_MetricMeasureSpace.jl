import Test: @test, @test_throws, @testset
import ObjectMatching: MetricMeasureSpace, DiscreteProbability, NormalizedPositiveVector


@testset "MetricMeasureSpace" begin
    dim::Int64 = 10
    C          = rand(Float64, (2,2))
    μ          = rand(Float64, 2)
    M          = MetricMeasureSpace(C, μ)

    # negative entries in mu raise error 
    @test_throws ArgumentError MetricMeasureSpace(C, Float64[-1, 1])

    # once constructed, mu sums to 1
    @test sum(M.μ)≈1

    function mu_is_constant_when_default(C)
        mu_mms = MetricMeasureSpace(C).μ
        return all(mu_mms .== mu_mms[1])
    end
    @test mu_is_constant_when_default(C)

    function mu_is_constant_using_outer_constructor(M::MetricMeasureSpace)
        mu_mms = M.μ
        return all(mu_mms .== mu_mms[1])
    end
    @test mu_is_constant_using_outer_constructor(MetricMeasureSpace((x,y)->abs(x-y), rand(4)))
    @test mu_is_constant_using_outer_constructor(MetricMeasureSpace((x,y)->abs(x-y), rand(4,2)))


    # raise error when C and mu have different dimensions 
    @test_throws ArgumentError MetricMeasureSpace(C, Float64[1, 1, 1])

    # raise error when C is not square 
    @test_throws ArgumentError MetricMeasureSpace(Float64[[1] [1]])
    @test_throws ArgumentError MetricMeasureSpace(Float64[[1, 1, 1] [1, 1, 1]])

    # raise error if \mu is always zero
    @test_throws ArgumentError MetricMeasureSpace(C, zeros(Float64, dim))
end 
    
