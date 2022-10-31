using Graphs
using Plots, ColorSchemes
import Statistics: mean
include("../mpdbp.jl")
include("../exact/montecarlo.jl")

T = 3
q = q_glauber

J = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0] .|> float
for ij in eachindex(J)
    if J[ij] !=0 
        J[ij] = randn()
    end
end
J = J + J'
gd = IndexedBiDiGraph(J)
g = IndexedGraph(J)

N = 5
h = randn(N)

β = 1.0

p⁰ = map(1:N) do i
    r = rand()
    r = 0.15
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]

O = [ (1, 2, 1, [0.1 0.9; 0.3 0.4]),
      (3, 4, 2, [0.4 0.6; 0.5 0.9]),
      (3, 5, 2, rand(2,2)) ,
      (2, 3, T, rand(2,2))]

ψ = pair_observations_directed(O, gd, T, q)
ψ_nondirected = pair_observations_nondirected(O, g, T, q)

# ψ = [[ones(q,q) for t in 1:T] for _ in edges(g)]

ising = Ising(J, h, β)
gl = Glauber(ising, p⁰, ϕ, ψ_nondirected)

ε = 1e-5
bp = mpdbp(ising, T, ϕ, ψ, p⁰)
cb = CB_BP(bp)
svd_trunc = TruncThresh(ε)
iterate!(bp, maxiter=10; svd_trunc, cb)
println()
@show cb.Δs

b = beliefs(bp)

@show m_bp = magnetizations(bp)

p = exact_prob(gl)
m = site_marginals(gl; p)
mm = site_time_marginals(gl; m)
m_exact = site_time_magnetizations(gl; mm)

cg = cgrad(:matter, N, categorical=true)
pl = plot(xlabel="BP", ylabel="exact", title="Magnetizations")
for i in 1:N
    scatter!(pl, m_bp[i], m_exact[i], c=cg[i], label="i=$i")
end
plot!(pl, identity, ls=:dash, la=0.5, label="", legend=:outertopright)

sms = sample(bp, 10^6)
b_mc = marginals(sms)
m_mc = [[bbb[1]-bbb[2] for bbb in bb] for bb in b_mc]

cg = cgrad(:matter, N, categorical=true)
pl = plot(xlabel="BP", ylabel="Monte Carlo", title="Magnetizations")
for i in 1:N
    scatter!(pl, m_bp[i], m_mc[i], c=cg[i], label="i=$i")
end
plot!(pl, identity, ls=:dash, la=0.5, label="", legend=:outertopright)