using Graphs
using Plots, ColorSchemes
import Statistics: mean
include("../mpdbp.jl")
include("../exact/montecarlo.jl")

q = q_glauber
T = 3

J = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0] .|> float
N = 5
h = randn(N)

β = 1.0

p⁰ = map(1:N) do i
    r = rand()
    r = 0.15
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ϕ[1][1] = [1, 0]
ϕ[2][2] = [0, 1]
ϕ[2][3] = [0, 1]

ising = Ising(J, h, β)
gl = ExactGlauber(ising, p⁰, ϕ)
m = site_marginals(gl)
mm = site_time_marginals(gl; m)

ε = 1e-2
bp = mpdbp(ising, T, ϕ, p⁰)
cb = CB_BP(bp)
iterate!(bp, maxiter=10; ε, cb)
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


avg_mag = mean(m_exact)
pl2 = plot(0:T, avg_mag, m=:o, xlabel="time", ylabel="average mag", label="BP")

# gs = sample(ising, nsweeps=10^4)
# m_mc = mean(magnetizations(gs))
# hline!(pl2, [m_mc.val], ribbon=[m_mc.err], label="Monte Carlo @ equilibrium",
#     legend=:right, title="magnetization")