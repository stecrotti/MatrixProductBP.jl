using Test, MatrixProductBP
using MatrixProductBP.Models
using Graphs, IndexedGraphs, Random

include("mpem.jl")
include("normalizations.jl")
# include("glauber_small_tree.jl")
include("pair_observations.jl")
include("sis_small_tree.jl")
include("sirs_small_tree.jl")
include("infinite_graph.jl")

nothing