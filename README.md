# Barycenters in the Gromov-Wasserstein space

![figure interpolation](https://raw.githubusercontent.com/Willinki/GWObjectMatching.jl/main/docs/gw-barycenters.png)

This repository implements the object matching procedure illustrated in:

> Gabriel Peyré, Marco Cuturi, Justin Solomon, Gromov-Wasserstein Averaging of Kernel and Distance Matrices, Proc. of ICML 2016.

We implemented two possible application of the algorithm as a possible demo use case. See the `usage` section for details.

## Requirements

Packages and additional resources needed to run the project are contained in the `Project.toml` file. It is possible to create a virtual environment from its specifications using `Pkg`.

## Theory and implementation details

Everything can be consulted in the `docs` directory or directly from the source code docstrings. A proper documentation is not available yet. Sorry 😢.

## Usage

As stated above, we implemented to possible use cases for the algorithm: figure interpolation and MNIST barycenter.

Scripts for these two examples can be found in the `demo` directory. It is sufficient to activate the project (ObjectMatching) and run one of the following commands from a terminal:

```{bash}
julia -i figure_interpolation.jl <options>
```

```{bash}
julia -i mnist_barycenters.jl <options>
```
A detailed list of options and overall explanation of both scripts is given below. (Note however that, for optional arguments, the default values work well.)

### Figure interpolation

This scripts implements the procedure illustrated in the image below. Given two figures in the `data/shapes` directory, generates images to interpolate between the two. Below, every optional argument is listed:

* `fromshape_name: [String, def="annulus"]`. Starting image for interpolation. A name between the ones in `data/shapes` must be inserted.  
* `toshape_name: [String, def="heart2"]`. Last image for interpolation. A name between the ones in `data/shapes` must be inserted.
* `ninterpolating: [Int, def=3]`. Number of interpolating images between fromshape and toshape.
* `npoints: [Int, def=500]`. Sometimes images contain a lot of points. It it possible to perform undersampling to speed up calculations. Here we specify the number of points that we want to keep in each image. They are sampled randomly (we will introduce Voronoi's partitions in the future).
* `SK_tol: [Float, def=1e-12]`. Tolerance for the stopping condition on the Sinkhorn Knopp algorithm. The smallest the tolerance, the larger the compute time. Consult documentation for details.
* `Ts_tol: [Float, def=0.001]`. Tolerance for the stopping condition on the transport matrix T. Same principle as above holds, the iteration will go on until each element in the matrix is stable up to this value.
* `Cp_niter: [Int, def=10]`. Number of iterations for the update of the final barycenter C. 
* `epsilon: [Float, def=0.05]`. Epsilon value for the entropic approximation of the OT problem. Changing this parameter might cause convergence problems, as stated in the paper. If the computation requires too much time, it might be useful to increase the value.
* `reconstruct_tol: [Float, def=1e-3]`. Tolerance for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.
*  `reconstruct_max_iter: [Int, def=2000]`. Number of interations for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.

### MNIST Barycenter

Given a number chosen between `[3, 4, 5, 8]` calculates their barycenter. Below, every optional argument is listed:

* `mnist_number: [Int, def=3]`. MNIST number to extract the barycenter from. The only possible choices are `[3, 4, 5, 8]`. See the directory `data/mnist` for more details. 
* `npoints: [Int, def=68]`. Sometimes images contain a lot of points. It it possible to perform undersampling to speed up calculations. Here we specify the number of points that we want to keep in each image. They are sampled randomly (we will introduce Voronoi's partitions in the future).
* `SK_tol: [Float, def=1e-12]`. Tolerance for the stopping condition on the Sinkhorn Knopp algorithm. The smallest the tolerance, the larger the compute time. Consult documentation for details.
* `Ts_tol: [Float, def=0.001]`. Tolerance for the stopping condition on the transport matrix T. Same principle as above holds, the iteration will go on until each element in the matrix is stable up to this value.
* `Cp_niter: [Int, def=10]`. Number of iterations for the update of the final barycenter C. 
* `epsilon: [Float, def=0.05]`. Epsilon value for the entropic approximation of the OT problem. Changing this parameter might cause convergence problems, as stated in the paper. If the computation requires too much time, it might be useful to increase the value.
* `reconstruct_tol: [Float, def=1e-3]`. Tolerance for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.
*  `reconstruct_max_iter: [Int, def=2000]`. Number of interations for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.

## Notes

This algorithm does not work yet, we are actively working on it. See the `debugging` branch for updates.
