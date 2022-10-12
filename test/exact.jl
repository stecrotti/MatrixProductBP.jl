using Test
using IndexedGraphs

include("../exact/exact_glauber.jl")

N = 5
T = 3
A = [0 1 0 1 1;
     1 0 1 1 0;
     0 1 0 0 1;
     1 1 0 0 0;
     1 0 1 0 0]
g = IndexedGraph(A)
J = ones(ne(g))
h = ones(nv(g))
J .= 0; h .= 0
β = 1.0
ising = Ising(g, J, h, β)


p⁰ = map(1:N) do i
    r = rand()
    r = 0.5
    [r, 1-r]
end
ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]

i = 3
t = 1
ϕ[i][t+1] = [0, 1]

gl = ExactGlauber(ising, p⁰, ϕ)
m = site_marginals(gl)