# degree 3
using Plots
include("../ensemble.jl")
include("../../telegram/notifications.jl")

unicodeplots()

T = 10
β = 1.0
wᵢ = fill( GlauberFactor(ones(3), 0.0, β), T )
ϕᵢ = [ones(2) for _ in 1:T]
pᵢ⁰ = [0.75, 0.25]

ε = 1e-2

A = rand_mpem2(2, T, d=1)
Δs = zeros(0)

A, iters, Δs = iterate_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; ε, maxiter=100, tol=1e-3, Δs)

ε = 2e-3
A, iters, Δs = iterate_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; ε, maxiter=100, tol=1e-3, Δs)

println("### Computing magnetization...\n")
mag = magnetization_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; ε)


p1 = plot(0:T, mag, label="mag", title="T=$T, β=$β", xlabel="time",
    ylabel="magnetiz")
p2 = plot(Δs, xlabel="time", ylabel="error")
p3 = plot(p1, p2, layout=(2,1))