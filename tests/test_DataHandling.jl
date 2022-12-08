import Test: @test, @testset, @test_throws 
import FileIO: load
import Distances: pairwise, euclidean
import ObjectMatching as OM
using Plots

# FIXTURES
N::Int64 = 2000
parameters = Dict([(:maxiter, 2000), (:tol,1e-1)])
img_path::String = "../src/data/shapes/heart.png"
img_path_gif::String = "../src/data/shapes/apple0.gif"
img_path_png::String = "../src/data/shapes/annulus.png"

function point_reconstruction(points::Matrix{Float64}; kwargs...)
    C::Matrix{Float64} = pairwise(euclidean, eachrow(points); symmetric=true)
    return OM.reconstruct_points(C; kwargs...)
end
# END OF FIXTURES

@testset "Data handling" begin
    function test_image_is_imported_as_points_and_normalized(img_path, N::Int64)
        img = load(img_path)
        points = (
            img |> OM.normalize_image
            |> OM.get_coord_black_points |> OM.rescale_convert_to_float
        )
        points = OM.undersample(points, N)
        dims = size(points, 1), size(points, 2)
        return points, dims
    end
    points, dims = test_image_is_imported_as_points_and_normalized(img_path, N) 
    @test dims[1] == N
    @test dims[2] == 2 
    @test OM.undersample(points, 0.5) isa Any
    @test OM.load_image(img_path_gif, n=500) isa Any
    @test OM.load_image(img_path_png, n=500) isa Any
    @test OM.load_image(img_path, n=500) isa Any
    @test size(point_reconstruction(points; parameters...)) == size(points)
    @test_throws ArgumentError OM.undersample(points, 1.1)
    @test_throws ArgumentError OM.undersample(points, -0.1)
    @test_throws ArgumentError OM.undersample(points, 0)
end

# the rest is simply a plot to check
points_rot::Matrix{Float64} = OM.load_image(img_path, n=N)
points_rec::Matrix{Float64} = point_reconstruction(points_rot; parameters...)
l = @layout [rotated reconstructed]
p2 = scatter(points_rot[:, 1], points_rot[:, 2], label="Rotated points");
p3 = scatter(points_rec[:, 1], points_rec[:, 2], label="Reconstructed");
plot(p2, p3, layout=l, display=true)
