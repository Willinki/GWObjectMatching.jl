module ObjectMatching

#using Distances
include("MetricMeasureSpace.jl")
include("loss.jl")
include("RepeatUntilConvergence.jl")
include("sinkhorn.jl")

end # module ObjectMatching
