using Graphs
using Plots
using LinearAlgebra
using Statistics
include("../mpdbp.jl")
include("../glauber.jl")

q = q_glauber
T = 5
N = 4
J = ones(N,N) - diagm(ones(N))
h = zeros(N)
β = 1.0
ising = Ising(J, h, β)

p⁰ = map(1:N) do i
    r = 0.75
    [r, 1-r]
end
ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]

ε = 1e-3
svd_trunc = TruncThresh(ε)
bp = mpdbp(ising, T; ϕ, p⁰, d=1)
cb = CB_BP(bp)

@time iterate!(bp, maxiter=10; svd_trunc, cb, tol=1e-3)

m_bp = cb.mag
pl = plot(0:T, mean(m_bp), xlabel="time", ylabel="magnetization", label="")
pl2 = plot(cb.Δs, xlabel="time", ylabel="Δ", label="")
plot(pl, pl2, layout=(2,1))