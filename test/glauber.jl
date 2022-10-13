using Graphs
using Plots, ColorSchemes
include("../mpdbp.jl")

q = q_glauber
T = 3

J = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0] .|> float
N = 5
h = randn(N)
β = 2.0

p⁰ = map(1:N) do i
    r = rand()
    # r = 0.4
    [r, 1-r]
end
ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]
ϕ[1][1] = [1, 0]
ϕ[2][2] = [0, 1]
ϕ[2][3] = [0, 1]

ising = Ising(J, h, β)
gl = ExactGlauber(ising, p⁰, ϕ)
m = site_marginals(gl)
mm = site_time_marginals(gl; m)

# εs = [0.0, 1e-2, 1e-1, 0.5]
# pls = [plot() for _ in εs]

# pls = map(eachindex(εs)) do i
#     ε = εs[i]
#     bp = mpdbp(gl)
#     cb = CB_BP(bp)
#     iterate!(bp, maxiter=5; ε, cb)
#     println()
#     @show cb.Δs

#     b = beliefs(bp; ε)
#     @show m_bp = magnetizations(bp)
#     m_exact = site_time_magnetizations(gl)

#     cg = cgrad(:matter, N, categorical=true)
#     pl = plot(xlabel="BP", ylabel="exact", title="Magnetizations")
#     for i in 1:N
#         scatter!(pl, m_bp[i], m_exact[i], c=cg[i], label="i=$i")
#     end

#     plot!(pl, identity, ls=:dash, la=0.5, label="", legend=:outertopright)
# end

ε = 0.1
bp = mpdbp(gl)
cb = CB_BP(bp)
iterate!(bp, maxiter=5; ε, cb)
println()
@show cb.Δs

b = beliefs(bp; ε)

@show m_bp = magnetizations(bp; ε)
m_exact = site_time_magnetizations(gl; m, mm)

cg = cgrad(:matter, N, categorical=true)
pl = plot(xlabel="BP", ylabel="exact", title="Magnetizations")
for i in 1:N
    scatter!(pl, m_bp[i], m_exact[i], c=cg[i], label="i=$i")
end

plot!(pl, identity, ls=:dash, la=0.5, label="", legend=:outertopright)


# T = 4
# N = 3
# A = [ mpem2(q, T; d=3) for _ in 1:3]
# pᵢ⁰ = [0.5, 0.5]
# wᵢ = [ GlauberFactor(ones(3), 0.0) for _ in 1:T]
# ϕᵢ = [[0.5, 0.5] for _ in 1:T]

