using MatrixProductBP, MatrixProductBP.Models

T = 100
J = 0.8
β = 1.0
h = 0.05
k = 3
M = 3
wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T+1)
bp = periodic_mpbp_infinite_graph(k, wᵢ, 2)
cb = CB_BP(bp);

svd_trunc=TruncBond(M)

include("../../telegram/notifications.jl")
# @telegram "glauber periodic infinite" begin
    iters, cb = iterate!(bp; maxiter=50, svd_trunc, cb, tol=1e-5, damp=0.0)
# end

using Plots
unicodeplots()

pl_conv = plot(cb.Δs, ylabel="convergence error", xlabel="iters", yaxis=:log10, legend=:outertopright,
    size=(300,200))
display(pl_conv)

spin(x, i) = 3-2x
spin(x) = spin(x, 0)
m = only(means(spin, bp))

m_static = equilibrium_observables(RandomRegular(k), J; β, h)[:m]
pl = scatter(m, label="MPBP")
plot!(pl, 1:T+1, fill(m_static, T+1), label="equilibrium",
    ylims=m_static .+ 1e-1 .* (-1, 1))
display(pl)