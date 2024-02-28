# Matrix Product Belief Propagation

[![CI](https://github.com/stecrotti/MatrixProductBP/actions/workflows/ci.yml/badge.svg)](https://github.com/stecrotti/MatrixProductBP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/stecrotti/MatrixProductBP/branch/main/graph/badge.svg?token=X30C323BYT)](https://codecov.io/gh/stecrotti/MatrixProductBP)

This repository contains the code relative to the paper _Matrix Product Belief Propagation for reweighted stochastic dynamics over graphs_ ([PNAS](https://www.pnas.org/doi/10.1073/pnas.2307935120), [arxiv](https://arxiv.org/abs/2303.17403)). 
When possible, variable names match the notation used there. 
An [errata](errata.md) file is also found here.

### Installation
Usage requires Julia version >= 1.8, although 1.9 or 1.10 are recommended. You can download Julia [here](https://julialang.org/downloads/).

To install the package:
```julia
using Pkg; Pkg.add("MatrixProductBP")
```

## Overview
The package is based on [TensorTrains.jl](https://github.com/stecrotti/TensorTrains.jl) for the operations on tensor trains (a.k.a. matrix product states).

The directory is organized as follows:
- The main module `MatrixProductBP` implements the BP equations.
    Essential ingredients are:
    - An `IndexedBiDiGraph` on which the system is defined  (for more details, see the [documentation of IndexedGraphs.jl](https://stecrotti.github.io/IndexedGraphs.jl/stable/bidigraph/))
    - `BPFactor`s living on the factor nodes of the graph. Any concrete subtype (e.g. `SISFactor`) must implement a method `(fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, xᵢᵗ::Integer)` that computes the contribution of that factor to the joint probability as a function of $x_i^{t+1},\boldsymbol{x}_{\partial i}^{t}, x_i^{t}$.
    - Optionally, single-node reweightings `ϕ` and edge reweightings `ψ`
    
    Some allowed operations:
    - `iterate!(bp::MPBP; kw...)`
    - `pair_beliefs(bp::MPBP)`
    - `beliefs(bp::MPBP)`
    - `bethe_free_energy(bp::MPBP)`
    
    There is some montecarlo (Soft Margin) utility in `src/sampling/`:
    - `sample(bp::MPBP, nsamples::Integer)`: draw samples from the prior weighted with their likelihood
    - `draw_node_observations!(bp::MPBP, nobs::Integer)`: draw samples from the prior and add them to the `MPBP` object as reweighting
    
    Code for computing probabilities exactly by exhaustive enumeration in `src/exact/` (only for small system size and final time):
    - `exact_prob(bp::MPBP)`
    
- A submodule `Models` that can be included with 
```julia
using MatrixProductBP.Models
```
that implements two models: Glauber dynamics and the SIS model of epidemic spreading.
It contains model definitions and specialized versions of the BP equations that are more efficient than the generic one.

- A `test` folder with small examples 

- A `notebooks` folder where results and plots are produced. The notebooks generating the results contained in the article output generate some data in .jld format that is then read by `notebooks/plots_for_paper.ipynb` which produces the figures in the article.

## Quickstart
To get a hang of how the package works, take a look at, e.g. `test/glauber_small_tree.jl`, `notebooks/glauber_infinite_graph_large_degree_small_beta.ipynb`, or `notebooks/sis_compare_softmargin.ipynb`.
