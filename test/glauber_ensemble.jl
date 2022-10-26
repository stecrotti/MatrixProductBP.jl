# degree 3
using Plots, DelimitedFiles
include("../ensemble.jl")
# include("../../telegram/notifications.jl")

unicodeplots()

T = 5
β = 1.0
wᵢ = fill( GlauberFactor(ones(3), 0.0, β), T )
ϕᵢ = [ones(2) for _ in 1:T]
pᵢ⁰ = [0.75, 0.25]

svd_trunc = TruncThresh(1e-3)

A = rand_mpem2(2, T, d=1)
Δs = zeros(0)

A, iters, Δs = iterate_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; svd_trunc, maxiter=100, 
    tol=1e-3, Δs)
p2 = plot(Δs, xlabel="iter", ylabel="error")
println("### Computing magnetization...\n")
mag = magnetization_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; svd_trunc)
mag_fast = magnetization_rs(A)

ff = readdlm("/home/crotti/MPdBP/nbs/montecarlo_N1000.txt", Float64)
mag_mc = ff[1:T+1,2]

[mag mag_fast mag_mc]


# p1 = plot(0:T, mag, label="MPdBP", title="T=$T, β=$β", xlabel="time",
#     ylabel="magnetiz")
# 
# plot!(p1, ff[1:T+1,1], ff[1:T+1,2], label="Monte Carlo", m=:diamond)
# p3 = plot(p1, p2, layout=(2,1))