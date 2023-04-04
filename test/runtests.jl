using Test, MatrixProductBP
using MatrixProductBP.Models
using Graphs, IndexedGraphs, Random, Distributions

include("equilibrium.jl")
# include("glauber_infinite_graph.jl")
include("glauber_pmJ_small_tree.jl")
include("glauber_small_tree.jl")
include("mpem.jl")
include("normalizations.jl")
include("pair_observations.jl")
include("sampling.jl")
include("sirs_small_tree.jl")
# include("sis_infinite_graph.jl")
# include("sis_small_tree.jl")

nothing