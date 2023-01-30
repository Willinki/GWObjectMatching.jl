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

