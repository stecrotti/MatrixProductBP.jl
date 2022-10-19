using Graphs
using Plots, ColorSchemes
import Statistics: mean
include("../mpdbp.jl")
include("../exact/montecarlo.jl")

q = q_glauber
T = 5

# J = [0 1 0 0 0;
#      1 0 1 0 0;
#      0 1 0 1 1;
#      0 0 1 0 0;
#      0 0 1 0 0] .|> float
# for ij in eachindex(J)
#     if J[ij] !=0 
#         J[ij] = randn()
#     end
# end
# J = J + J'
# N = 5

# h = randn(N)
# β = 1.0
# ising = Ising(J, h, β)

N = 10
k = 3
gg = random_regular_graph(N, k)
g = IndexedGraph(gg)
h = zeros(N)
J = ones(ne(g))
β = 1.0
ising = Ising(g, J, h, β)

p⁰ = map(1:N) do i
    r = rand()
    r = 0.75
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]

gl = Glauber(ising, p⁰, ϕ)

ε = 1e-2
bp = mpdbp(gl, d=1)
cb = CB_BP(bp)
iterate!(bp, maxiter=50; ε, cb, tol=1e-3)
println()
@show cb.Δs

b = beliefs(bp; ε)
m_bp = magnetizations(bp; ε)
avg_mag = mean(m_bp)
pl = plot(0:T, avg_mag, m=:o, xlabel="time", ylabel="average mag", label="BP")

gs = sample(gl, nsamples=10^4, showprogress=true)
m = magnetizations(gs)
mm = mean(m, dims=2) |> vec
plot!(pl, 0:T, mm, label="Monte Carlo", m=:square)