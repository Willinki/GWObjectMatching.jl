"""
This is a demo script, given a number chosen between 3,4,5,8, it calculates the
barycenter between all the figures inside the directory data/mnist/<number>
"""


import ObjectMatching as OM
import Distances: euclidean
using PartialFunctions
using ArgParse
using Plots

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
	"--mnist_number"
            help="Perform the experiment on images depicting the number n"
            arg_type=Int64
            range_tester=x->x in [3, 4, 5, 8]
            default=8
        "--npoints"
            help="Number of points for undersampling"
            arg_type=Int64
            default=65
        "--SK_tol"
            help="Tolerance for the stopping condition on SK algorithm"
            arg_type=Float64
            default=1e-8
        "--Ts_tol"
            help="Tolerance for the stopping condition on Ts"
            arg_type=Float64
            default=1e-6
        "--Cp_niter"
            help="Number of iterations for Cp updates"
            arg_type=Int64
            default=20
        "--epsilon"
            help="Epsilon value for the entropic approximation of the OT problem"
            arg_type=Float64
            default=0.0010
        "--reconstruct_tol"
            help="Tolerance for points reconstruction algorithm"
            arg_type=Float64
            default=1e-3
        "--reconstruct_max_iter"
            help="Max number of iterations for points reconstruction algorithm"
            arg_type=Int64
            default=2000
    end
    return parse_args(s)
end

"""
Given an integer numbers, returns an array of paths of each file corresponding to the
chosen number (only 3,4,5,8 are admissible). Image files can be found in data/mnist
"""
function list_img_paths(number::Int64)::Vector{String}
    dir::String = joinpath(OM.HOME_DIR, "data", "mnist", string(number))
    return readdir(dir, join=true)
end

function plot_results(
        img_list::Vector{Matrix{Float64}},
        barycenter::Matrix{Float64},
        outfile=joinpath(OM.HOME_DIR, "demo", "plot_numbers.png")
    )
    img_plots = [
        scatter(
            x[:, 1], x[:, 2],
            aspect_ratio=:equal,
            color="orange",
            markersize=1.5,
            markerstrokecolor="orange",
            markeralpha=2
        )
        for x in img_list
    ]
    barycenter_plot = scatter(
        barycenter[:, 1], barycenter[:, 2],
        aspect_ratio=:equal, 
        color="blue",
        markersize=1.5,
        markerstrokecolor="blue",
        markeralpha=2
    )
    plot(
        img_plots..., barycenter_plot;
        axis=([], false), legend=false, plot_title="Mnist Barycenter - demo",
        dpi=300
    )
    savefig(outfile)
    println("File saved to $outfile")
end

function main()
    args::Dict = parse_commandline()
    files_paths::Vector{String} = list_img_paths(args["mnist_number"])
    images_list::Vector{Matrix{Float64}} = map(
        OM.load_image $ (n=args["npoints"],), 
        files_paths
    )
    images_MMS::Vector{OM.MetricMeasureSpace} = [
        OM.MetricMeasureSpace(euclidean, image) for image in images_list
    ]
    reconstruction_pars = Dict([
        (:maxiter,  args["reconstruct_max_iter"]),
        (:tol,      args["reconstruct_tol"]),
    ])
    barycenters_pars = Dict([
        (:n_points,  args["npoints"]),
        (:ϵ,         args["epsilon"]),
        (:Cp_niter,  args["Cp_niter"]),
        (:Ts_tol,    args["Ts_tol"]),
        (:SK_tol,    args["SK_tol"]),
    ])
    @info "Computing barycenter"
    barycenter_dist = OM.GW_barycenters(images_MMS; barycenters_pars...)
    barycenter_img  = OM.reconstruct_points(barycenter_dist.C; reconstruction_pars...)
    plot_results(images_list, barycenter_img)
end

main()

