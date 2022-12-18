import ObjectMatching as OM
using Plots
using ArgParse
using LinearAlgebra
import Distances: euclidean, pairwise

function list_img_paths(number::Int64)::Vector{String}
    dir::String = joinpath(OM.HOME_DIR, "data", "mnist", string(number))
    return readdir(dir, join=true)
end

function initialize_C(p::Vector{Float64})
    #WLOG we can assume that p.>0, if it is not we discard its zero elements
    p = filter!(e->e!=0, p) 
    # initialize C uniform
    return C = OM.MetricMeasureSpace(rand(Float64, length(p), length(p)), p)
end

function stop_bar_niter(history::Vector{OM.MetricMeasureSpace})
    return length(history)>=niter
end

function main()

end

main()
