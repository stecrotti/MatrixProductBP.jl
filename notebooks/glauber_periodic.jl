# import Pkg; Pkg.develop(path="../../.julia/dev/BeliefPropagation/")
using MatrixProductBP, MatrixProductBP.Models
using Statistics
using Graphs, IndexedGraphs, Random

### LOOPY RANDOM GRAPH
T = 15
J = 0.8
β = 1.0
h = 0.2
rng = MersenneTwister(0)
N = 4
k = 3
g = random_regular_graph(N, k) |> IndexedGraph
ising = Ising(J * adjacency_matrix(g), h*ones(nv(g)), β)
gl = Glauber(ising, T)
M = 15
bp = periodic_mpbp(gl; d=M)
cb = CB_BP(bp)

svd_trunc=TruncBond(M)
iters, cb = iterate!(bp; maxiter=2, svd_trunc, cb, tol=1e-5, damp=0.0)


# import BeliefPropagation
# bp_static = BeliefPropagation.BP(BeliefPropagation.Models.Ising(J * adjacency_matrix(g), h*ones(nv(g)), β))
# BeliefPropagation.iterate!(bp_static; maxiter=100)
# m_static = reduce.(-, BeliefPropagation.beliefs(bp_static)) |> mean

# include("../../telegram/notifications.jl")
# @telegram "glauber periodic"

# using Plots

# unicodeplots()
# pl_conv = plot(cb.Δs, ylabel="convergence error", xlabel="iters", yaxis=:log10, legend=:outertopright,
#     size=(300,200))
# display(pl_conv)

# spin(x, i) = 3-2x
# spin(x) = spin(x, 0)
# m = mean(means(spin, bp))

# pl = scatter(m, label="MPBP")
# plot!(pl, 1:T+1, fill(m_static, T+1), label="equilibrium", ylims=m_static .+ 1e-1 .* (-1, 1))
# display(pl)

import TensorTrains
TensorTrains.bond_dims.(bp.μ) |> display