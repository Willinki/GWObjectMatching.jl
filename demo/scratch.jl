import ObjectMatching as OM
using Plots
using ArgParse
import Distances: euclidean, pairwise
import LinearAlgebra: diag

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"--mnist_number", "-n"
            help="Perform the experiment on images depicting the number n"
            arg_type=Int64
            range_tester=x->x in [3, 4, 5, 8]
            default=3
        "--barycenter_T_tol"
            help="Tolerance for the stopping condition on optimal transport"
            arg_type=Float64
            default=1e-6
        "--barycenter_ab_tol"
            help="Tolerance for the stopping condition on marginals"
            arg_type=Float64
            default=1e-6
        "--reconstruct_tol"
            help="Tolerance for points reconstruction algorithm"
            arg_type=Float64
            default=1e-2
        "--reconstruct_max_iter"
            help="Max number of iterations for points reconstruction algorithm"
            arg_type=Int64
            default=5000
    end
    return parse_args(s)
end

function list_img_paths(number::Int64)::Vector{String}
    dir::String = joinpath(OM.HOME_DIR, "data", "mnist", string(number))
    return readdir(dir, join=true)
end

function plot_results(img_list::Vector{Matrix{Float64}}, barycenter::Matrix{Float64})
    list_plots = [scatter(x[:, 1], x[:, 2]) for x in img_list]
    barycenter_plot = scatter(barycenter[:, 1], barycenter[:, 2])
    plot(list_plots..., barycenter_plot)
    gui()
end

function main()
    args::Dict                                = parse_commandline()
    files_paths::Vector{String}               = list_img_paths(args["mnist_number"])
    images_list::Vector{Matrix{Float64}}      = map(OM.load_image, files_paths)
    n_points::Int64                           = 100 
    images_MMS::Vector{OM.MetricMeasureSpace} = [
        OM.MetricMeasureSpace(euclidean, image) for image in images_list
    ]
    reconstruction_pars = Dict([
        (:maxiter,  args["reconstruct_max_iter"]),
        (:tol,      args["reconstruct_tol"]),
    ])
    @info "Computing Barycenter"
    barycenter_dist::OM.MetricMeasureSpace = OM.GW_barycenters(n_points, images_MMS)
    println(diag(barycenter_dist.C))
    #@info "Reconstructing image"
    #barycenter_points::Matrix{Float64} = OM.reconstruct_points(
    #    barycenter_dist.C;
    #    reconstruction_pars...
    #)
    #plot_results(images_list, barycenter_points)
end

main()
