"""
This script simply runs all the individual tests.
"""
mytests = [
    "test_MetricMeasureSpace.jl"
    "test_RepeatUntilConvergence.jl"
]
println("Running tests...")
for test in mytests
    include(test)
end
