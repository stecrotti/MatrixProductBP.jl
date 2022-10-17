using Graphs
using Plots, ColorSchemes
include("../mpdbp.jl")

q = q_glauber
T = 4
N = 10
k = 3
h = zeros(N)

gg = random_regular_graph(N, k)
g = IndexedGraph(gg)
J = ones(ne(g))
β = 1.0

p⁰ = map(1:N) do i
    # r = rand()
    r = 1e-3
    [r, 1-r]
end
ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]

ising = Ising(g, J, h, β)
ε = 1e-1
bp = mpdbp(ising, T, ϕ, p⁰)
cb = CB_BP(bp)
iterate!(bp, maxiter=5; ε, cb)
println()
@show cb.Δs

m_bp = cb.mag