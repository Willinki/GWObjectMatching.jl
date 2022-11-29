import Test: @test, @test_throws
import .ObjectMatching: MetricMeasureSpace

M = MetricMeasureSpace(ones(Float64, (2,2)), rand(Float64, 2))

@test_throws ArgumentError MetricMeasureSpace(ones(Float64, (2,2)), Float64[-1, 1])

@test sum(M.mu)==1
