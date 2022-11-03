using Graphs
using Plots, ColorSchemes
import Statistics: mean
include("../mpdbp.jl")
include("../glauber.jl")
include("../exact/montecarlo.jl")

q = q_glauber
T = 2

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

N = 5
h = randn(N)

β = 1.0

p⁰ = map(1:N) do i
    r = rand()
    r = 0.15
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ising = Ising(J, h, β)
ψ = [[ones(2,2) for t in 1:T] for _ in 1:ne(ising.g)]
gl = Glauber(ising, p⁰, ϕ, ψ)

ε = 1e-8
bp = mpdbp(ising, T; ϕ, p⁰)
draw_node_observations!(bp, 5)
cb = CB_BP(bp)
svd_trunc = TruncThresh(ε)
iterate!(bp, maxiter=10; svd_trunc, cb)
println()

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(gl)
b_exact = site_time_marginals(gl; m = site_marginals(gl; p=p_exact))
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]


pl_ex = plot(xlabel="time", ylabel="\$P(x_i=I)\$", xticks=0:5:T, title="Exact")
for i in 1:N
    plot!(pl_ex, 0:T, p_ex[i], label="i=$i", m=:o, ms=3, lw=1)
end
pl_bp = plot(xlabel="time", ylabel="\$P(x_i=I)\$", xticks=0:5:T, title="MPdBP")
for i in 1:N
    plot!(pl_bp, 0:T, p_bp[i], label="i=$i", m=:o, ms=3, lw=1)
end
pl_sc = scatter(reduce(vcat, p_bp), reduce(vcat, p_ex), xlabel="MPdBP", 
    ylabel="Exact", 
    label="\$P(x_i=I)\$", ms=3, c=:black, legend=:outertopright)
plot!(identity, label="", size=(300,300))
plot(pl_ex, pl_bp, pl_sc, titlefontsize=10, size=(950, 300), legend=:outertopright, 
    margin=5Plots.mm, layout=(1,3)) |> display

@show Z_exact exp(-cb.f[end]);