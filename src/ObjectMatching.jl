module ObjectMatching
include("Utils.jl")
HOME_DIR = return_home_dir()
include("DataHandling.jl")
include("MetricMeasureSpace.jl")
include("loss.jl")
include("RepeatUntilConvergence.jl")
include("SinkhornKnopp.jl")
include("Barycenters.jl")
end # module ObjectMatching
