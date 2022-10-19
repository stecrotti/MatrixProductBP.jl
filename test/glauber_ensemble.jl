# degree 3
using Plots
include("../ensemble.jl")
include("../../telegram/notifications.jl")

unicodeplots()

T = 20
β = 1.8
wᵢ = fill( GlauberFactor(ones(2), 0.0, β), T )
ϕᵢ = [ones(2) for _ in 1:T]
pᵢ⁰ = [0.75, 0.25]

ε = 3e-2

A = rand_mpem2(2, T, d=1)
Δs = zeros(0)

A, iters, Δs, mag = @telegram begin
    A, iters, Δs = iterate_rs_deg3(A, pᵢ⁰, ϕᵢ; ε, maxiter=100, tol=1e-2, Δs)
    println("### Computing magnetization...\n")
    mag = magnetization_rs_deg3(A, pᵢ⁰, ϕᵢ; ε)
    A, iters, Δs, mag
end


p1 = plot(0:T, mag, label="mag", title="T=$T, β=$β", xlabel="time",
    ylabel="magnetiz")
p2 = plot(1:iters, Δs[1:iters], xlabel="time", ylabel="error")
p3 = plot(p1, p2, layout=(2,1))