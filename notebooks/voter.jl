using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs
using Statistics
using Plots, ColorSchemes, LaTeXStrings

T = 10
N = 10^2
k = 3
nsamples = 10^3

# seed = 7111
seed = 1
gg = erdos_renyi(N, k/N; seed)
g = IndexedGraph(gg)

w = map(1:N) do i
    ei = inedges(g, i)
    J = 1
    fill(HomogeneousVoterFactor(J), T+1)
    # fill(HomogeneousGlauberFactor(J, 0), T+1)
end

m0 = [0.7, 0.8, 0.9] |> reverse

samplers = map(eachindex(m0)) do a
    m = m0[a]
    bp = mpbp(IndexedBiDiGraph(gg), w, fill(2,N), T)
    for i in eachindex(bp.ϕ)
        bp.ϕ[i][begin] .= [(1+m)/2,(1-m)/2]
    end
    sms = sample(bp, nsamples; showprogress=false)
    println("Finished magnetiz $m ($a/$(length(m0)))")
    sms
end

spin(x, i) = 3 - 2x
spin(x) = spin(x, 0)
magnetiz(sms) = mean(getproperty.(mm, :val) for mm in means(spin, sms))
m = [magnetiz(sms) for sms in samplers]

# ff(x, i) = (3-2x) * degree(g, i)
# magnetiz(sms) = mean(getproperty.(mm, :val) for mm in means(ff, sms))
# m = [magnetiz(sms) for sms in samplers]

# fff(x, i) = (3-2x) * (degree(g, i) != 0)
# magnetiz(sms) = mean(getproperty.(mm, :val) for mm in means(fff, sms))
# m = [magnetiz(sms) for sms in samplers]

cg = cgrad(:matter, length(m0)+1, categorical=true)
plm = plot(xlabel=L"t", ylabel=L"m", legend=:outertopright)
for a in eachindex(samplers)
    plot!(plm, 0:T, m[a], label="m⁰=$(m0[a])", c=cg[a+1], m=:o, msc=:auto)
end
plm

c = map(samplers) do sms
    pm = pair_marginals_alternate(sms; showprogress=true)
    ee = map(pm) do bij
        expectation.(spin, bij)
    end
    mean(ee)
end

cg = cgrad(:matter, length(m0)+1, categorical=true)
ple = plot(xlabel=L"t", ylabel=L"\langle \sigma_i\sigma_j\rangle", legend=:outertopright)
for a in eachindex(samplers)
    plot!(ple, 1:T, c[a], label="m⁰=$(m0[a])", c=cg[a+1], m=:o, msc=:auto)
end
ple

plot(plm, ple, size=(800,400))

########## MPBP

sms = samplers[1]
bp = deepcopy(sms.bp)

cb = CB_BP(bp; showprogress=true)
maxiter = 10
svd_trunc = TruncBond(10)
tol = 1e-5
iterate!(bp; maxiter, svd_trunc, cb, tol)