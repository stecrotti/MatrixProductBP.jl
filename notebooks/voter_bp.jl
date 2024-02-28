using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs
using Statistics
using Plots

T = 50
N = 10^2
k = 3
m0 = 0.7

seed = 1
gg = erdos_renyi(N, k/N; seed)
g = IndexedGraph(gg)
w = fill(fill(HomogeneousVoterFactor(1), T+1), N)
bp = mpbp(IndexedBiDiGraph(gg), w, fill(2,N), T)
for i in eachindex(bp.ϕ)
    bp.ϕ[i][begin] .= [(1+m0)/2,(1-m0)/2]
end

#### montecarlo
nsamples = 10^3
sms = sample(bp, nsamples; showprogress=false)

spin(x, i) = 3 - 2x
spin(x) = spin(x, 0)
m_mc = mean(getproperty.(mm, :val) for mm in means(spin, sms))

pm = pair_marginals_alternate(sms; showprogress=true)
ee = map(pm) do bij
    expectation.(spin, bij)
end
c_mc = mean(ee)


########## MPBP

unicodeplots()

cb = CB_BP(bp)
tol = 1e-5
matrix_sizes = [5, 10, 15]
maxiters = fill(20, length(matrix_sizes))
iters = zeros(Int, length(maxiters))
for i in eachindex(maxiters)
    iters[i], _ = iterate!(bp; maxiter=maxiters[i], svd_trunc=TruncBond(matrix_sizes[i]), cb, tol)
end

iters_cum = cumsum(iters)
inds = 1:iters_cum[1]
pl = plot(inds, cb.Δs[inds], label="$(matrix_sizes[1])x$(matrix_sizes[1]) matrices")
for i in 2:length(iters)
    inds = iters_cum[i-1]:iters_cum[i]
   plot!(pl, inds, cb.Δs[inds], label="$(matrix_sizes[i])x$(matrix_sizes[i]) matrices")
end
plot(pl, ylabel="convergence error", xlabel="iters", yaxis=:log10, size=(500,300), legend=:outertopright)

m_bp = mean(means(spin, bp))

blue = theme_palette(:auto)[1]
pl = plot(xlabel="time", ylabel="magnetization")
plot!(pl, 0:T, m_mc, label="MonteCarlo", c=:black, m=:diamond, ms=3, msc=:auto, st=:scatter)
plot!(pl, 0:T, m_bp, label="MPBP",
    size=(500,300), ms=3, titlefontsize=12,
    legend=:bottomright, msc=:auto, c=blue, lw=2)
savefig(pl, "voter_bp.pdf")