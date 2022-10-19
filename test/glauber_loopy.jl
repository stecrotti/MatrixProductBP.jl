using Graphs
using Plots, ColorSchemes
using LinearAlgebra
using Statistics
include("../mpdbp.jl")

q = q_glauber
T = 10
N = 4
J = ones(N,N) - diagm(ones(N))
h = zeros(N)
β = 1.0
ising = Ising(J, h, β)

p⁰ = map(1:N) do i
    # r = rand()
    r = 0.75
    [r, 1-r]
end
ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]

ε = 1e-2
bp = mpdbp(ising, T, ϕ, p⁰, d=1)
cb = CB_BP(bp)
iterate!(bp, maxiter=10; ε, cb, tol=1e-3)

m_bp = cb.mag
pl = plot(0:T, mean(m_bp), xlabel="time", ylabel="magnetization", label="",
    m=:o)