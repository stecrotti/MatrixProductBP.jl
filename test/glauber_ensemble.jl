# degree 3
using Plots
include("../ensemble.jl")

unicodeplots()

T = 10
β = 1.0
wᵢ = fill( GlauberFactor(ones(2), 0.0, β), T )
ϕᵢ = [ones(2) for _ in 1:T]
pᵢ⁰ = [0.75, 0.25]

A = rand_mpem2(2, T, d=2)


iters, mag, Δs = iterate_rs_deg3(A, pᵢ⁰, ϕᵢ; ε=1e-2, maxiter=20, tol=1e-6)
p1 = plot(0:T, mag, label="mag", m=:o)
p2 = plot(1:iters, Δs[1:iters], m=:o)
p3 = plot(p1, p2, layout=(2,1))