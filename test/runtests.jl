using Test, MatrixProductBP
using MatrixProductBP.Models
using Graphs, IndexedGraphs, Random, Distributions

include("mpem.jl")
include("normalizations.jl")
include("glauber_small_tree.jl")
include("pair_observations.jl")
include("glauber_pmJ_small_tree.jl")
include("sis_small_tree.jl")
include("sirs_small_tree.jl")
include("sis_infinite_graph.jl")
include("sampling.jl")

nothing