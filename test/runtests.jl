using Test, MatrixProductBP
using MatrixProductBP.Models
using Graphs, IndexedGraphs, Random, Distributions, SparseArrays
using TensorTrains
using Aqua

@testset "Aqua" begin
    Aqua.test_all(MatrixProductBP, ambiguities=false)
    Aqua.test_ambiguities(MatrixProductBP)
end

include("equilibrium.jl")
include("glauber_infinite_graph.jl")
include("glauber_pmJ_small_tree.jl")
include("glauber_small_tree.jl")
include("mpems.jl")
include("normalizations.jl")
include("pair_observations.jl")
include("periodic.jl")
include("sampling.jl")
include("sirs_small_tree.jl")
include("sis_heterogeneous.jl")
include("sis_heterogeneous_compare_homogeneous.jl")
include("sis_infinite_graph.jl")
include("sis_small_tree.jl")

nothing