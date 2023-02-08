# Barycenters in the Gromov-Wasserstein space

![figure interpolation](https://raw.githubusercontent.com/Willinki/GWObjectMatching.jl/main/docs/gw-barycenters.png)

This repository implements the object matching procedure illustrated in:

> Gabriel Peyré, Marco Cuturi, Justin Solomon, Gromov-Wasserstein Averaging of Kernel and Distance Matrices, Proc. of ICML 2016.

We implemented two possible application of the algorithm as a possible demo use case. See the `usage` section for details.

## Requirements

Packages and additional resources needed to run the project are contained in the `Project.toml` file. It is possible to create a virtual environment from its specifications using `Pkg` and `instantiate`.

## Problem statement

The main purpose of the algorithm is to determine the barycenter of a given set of objects. Intuitively, the barycenter of a set of objects is another object with "intermediate shape"
between the two. 
This procedure is invariant with respect to transformation of these objects with isometries. In other words, the barycenter does between two objects does not change if we apply an isometry to one or both 
of the objects involved.
In our case, we consider figures as our objects (png images). The result (the barycenter) is an intermediate figure between two chosen. The algorithm, however, is very flexible in general.

Other than the simple barycenter calculation, which is performed in the `mnist_barycenter.jl` script, we present an additional application, which we consider the main achievement: figure interpolation.
An example of the procedure is illustrated in the image above.

### Image description

Given two figures (the leftmost and rightmost ones on each row of the image), we calculate their barycenter while varying the _weight_ with which each figure contributes, for a total four examples. 
The result is some form of _interpolation_ between figures, as visible in the image.

For each column, the fraction `t` represents the _weight_ of the second figure inside the computation of the barycenter. `t=0` implies that the second figure is completely absent, (the barycenter is equal to the first image).
The opposite holds for `t=1`, which corresponds to the second figure. In between the interpolation in action.

## Coding considerations

The main challenge of this project deals with efficiency, code clarity and flexibility. The algorithm is composed of three nested loops. In each of these loops, a quantity is updated repeatedly at each iteration, and the cycle is 
repeated until convergence. The authors of the paper left the concept of `convergence` undefined. We wrote the code to be easily extensible with respect to this. The structure `RepeatUntilConvergence` and its related method are able 
to standardize and make the whole process extensible, since the user can easily define new convergence conditions and integrate them seamlessly in the algorithm. 
This structure also allows us to get rid of explicit nested loop inside the code.

The downside of this is efficiency, in order for the structure to be general, we cannot optimize for the single use cases. The `execute!` method contains lots of deepcopies, some of which could be optimized.

In general, for every passage that is left ambiguos in the paper, we tried to structure the code to be extensible.

Other notable optimizations are:

* adding the possibility to perform operations on the GPU
* implement Voronoi partitions instead of random sampling for the subsampling of the images

## Theory and implementation details

For a formal description of the theory and implementation details of the procedure, see the `docs` directory.
The source code is also documented properly.
A proper documentation is not available yet. Sorry 😢.

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
* `epsilon: [Float, def=0.0003]`. Epsilon value for the entropic approximation of the OT problem. Changing this parameter might cause convergence problems, as stated in the paper. If the computation requires too much time, it might be useful to increase the value.
* `reconstruct_tol: [Float, def=1e-3]`. Tolerance for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.
*  `reconstruct_max_iter: [Int, def=2000]`. Number of interations for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.

### MNIST Barycenter

Given a number chosen between `[3, 4, 5, 8]` calculates their barycenter. Below, every optional argument is listed:

* `mnist_number: [Int, def=8]`. MNIST number to extract the barycenter from. The only possible choices are `[3, 4, 5, 8]`. See the directory `data/mnist` for more details. 
* `npoints: [Int, def=68]`. Sometimes images contain a lot of points. It it possible to perform undersampling to speed up calculations. Here we specify the number of points that we want to keep in each image. They are sampled randomly (we will introduce Voronoi's partitions in the future).
* `SK_tol: [Float, def=1e-12]`. Tolerance for the stopping condition on the Sinkhorn Knopp algorithm. The smallest the tolerance, the larger the compute time. Consult documentation for details.
* `Ts_tol: [Float, def=0.001]`. Tolerance for the stopping condition on the transport matrix T. Same principle as above holds, the iteration will go on until each element in the matrix is stable up to this value.
* `Cp_niter: [Int, def=10]`. Number of iterations for the update of the final barycenter C. 
* `epsilon: [Float, def=0.0002]`. Epsilon value for the entropic approximation of the OT problem. Changing this parameter might cause convergence problems, as stated in the paper. If the computation requires too much time, it might be useful to increase the value.
* `reconstruct_tol: [Float, def=1e-3]`. Tolerance for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.
*  `reconstruct_max_iter: [Int, def=2000]`. Number of interations for points reconstruction algorithm. It is used in order to reconstruct points from the distance matrix. See `LinearAlgebra.MetricMDS`.

## Notes

Times for the simulations are the following (on MacBook Pro M1 - 8GB): `figure_interpolation.jl: 5 min`, `mnist_barycenter.jl: 20min`. Times can be reduced at the expense of precision, simply by reducing `Ts_tolerance`, or the other similar parameters.

An example of results is in the `demo` directory.
