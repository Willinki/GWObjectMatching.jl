"""
This is a demo script.
Given two images in the data/shapes directory, it performs interpolation between the
two. It it possible to specify multiple options and customize the script with the
options below.
"""


import ObjectMatching as OM
import Distances: euclidean
using PartialFunctions
using Plots
using ArgParse
using ProgressMeter

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--fromshape_name"
            help="Starting image for interpolation"
            arg_type=String
            default="annulus"
        "--toshape_name"
            help="End image for interpolation"
            arg_type=String
            default="cross"
        "--ninterpolating"
            help="Number of interpolating images"
            arg_type=Int
            default=3
        "--npoints"
            help="Number of points for undersampling"
            arg_type=Int64
            default=200
        "--SK_tol"
            help="Tolerance for the stopping condition on SK algorithm"
            arg_type=Float64
            default=1e-6
        "--Ts_tol"
            help="Tolerance for the stopping condition on Ts"
            arg_type=Float64
            default=1e-6
        "--Cp_niter"
            help="Number of iterations for Cp updates"
            arg_type=Int64
            default=10
        "--epsilon"
            help="Epsilon value for the entropic approximation of the OT problem"
            arg_type=Float64
            default=0.0025
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
Given two strings representing filenames inside the folder data/shapes, returns
their full paths.

Returns
-------

* shapes_paths : Vector{String}
"""
function list_shapes_paths(fromshape::String, toshape::String)::Vector{String}
    dir::String                  = joinpath(OM.HOME_DIR, "data", "shapes")
    files_paths::Vector{String}  = readdir(dir, join=true)
    shapes_paths::Vector{String} = [
        filter(x -> occursin(shapename, x), files_paths) |> first
        for shapename in [fromshape, toshape] 
    ]
    return shapes_paths
end

"""
Given the initial images matrices and the calculated barycenter matrices, plots the
overall results.
"""
function plot_results(
        img_list::Vector{Matrix{Float64}},
        barycenters::Vector{Matrix{Float64}};
        outfile=joinpath(OM.HOME_DIR, "demo", "plot_figures.png")
    )
    img_plots = [
        scatter(
            x[:, 1], x[:, 2],
            aspect_ratio=:equal,
            color="blue",
            markersize=1.5,
            markerstrokecolor="blue",
            markeralpha=0.9
        )
        for x in img_list
    ]
    barycenter_plots = [
        scatter(
            barycenter[:, 1], barycenter[:, 2],
            aspect_ratio=:equal,
            color="orange",
            markersize=1.5,
            markerstrokecolor="orange",
            markeralpha=0.9
        )
        for barycenter in barycenters
    ]
    TOT_PLOT_NUMBER = length(img_plots)+length(barycenter_plots)
    plot(
        img_plots[1], barycenter_plots..., img_plots[end];
        layout=grid(1, TOT_PLOT_NUMBER), axis=([], false),
        legend=false, plot_title="Interpolation between images",
        dpi=300
    )
    savefig(outfile)
    println("File saved to $outfile")
end

function main()
    args::Dict = parse_commandline()
    files_paths::Vector{String} = list_shapes_paths(
        args["fromshape_name"], args["toshape_name"]
    )
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
    @info "Computing interpolation"
    interpolating_images = @showprogress [
        begin
            barycenter_dist = OM.GW_barycenters(
                images_MMS, OM.ConvexSum([1-lambda, lambda]);
                barycenters_pars...
            )
            OM.reconstruct_points(barycenter_dist.C; reconstruction_pars...)
        end
        for lambda in LinRange(0, 1, args["ninterpolating"]+2)[2:end-1]
    ]
    plot_results(images_list, interpolating_images)
end

main()
