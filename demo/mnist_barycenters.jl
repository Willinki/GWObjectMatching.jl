import ObjectMatching as OM
using ArgParse

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
            default=1e-3
        "--reconstruct_max_iter"
            help="Max number of iterations for points reconstruction algorithm"
            arg_type=Int64
            default=2000
    end
    return parse_args(s)
end

function list_img_paths(number::Int64)::Vector{String}
    dir::String = joinpath(OM.HOME_DIR, "data", string(number))
    return readdir(dir, join=true)
end

function plot_results(img_list::Vector{Matrix{Float64}}, barycenter::Matrix{Float64})
end

function main()
    args::Dict                           = parse_commandline()
    files_paths::Vector{String}          = list_img_paths(args[:mnist_number])
    images_list::Vector{Matrix{Float64}} = map(OM.load_images, files_paths)
    ## TODO:
    #barycenter_pars = Dict([
    #    (:ab_tol, args[:barycenter_ab_tol]),
    #    (:T_tol, args[:barycenter_T_tol]),
    #])
    # C::Matrix{Float64} = compute_barycenter(images_list, barycenter_pars...)#
    reconstruction_pars = Dict([
        (:max_iter, args[:reconstruct_max_iter]),
        (:tol,      args[:reconstruct_tol]),
    ])
    barycenter::Matrix{Float64} = OM.reconstruct_points(C; reconstruction_pars...)
    plot_results(images_list, barycenter)
end

main()

