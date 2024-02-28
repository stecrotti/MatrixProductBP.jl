using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs
using Statistics
using Plots, ColorSchemes, LaTeXStrings

Base.GC.gc()

T = 200
N = 10^2
k = 3
nsamples = 5*10^2

# seed = 7111
seed = 1
gg = erdos_renyi(N, k/N; seed)
g = IndexedGraph(gg)

Δt = 1
p0 = exp(-Δt)
p0 = 0

w = map(1:N) do i
    ei = inedges(g, i)
    J = 1
    fill(DampedFactor(HomogeneousVoterFactor(J), p0), T+1)
    # fill(HomogeneousGlauberFactor(J, 0), T+1)
end

m0 = [0.7, 0.8, 0.9] |> reverse
# m0 = [0.9]
m0 = [0.1]

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
    plot!(plm, 0:Δt:(T*Δt), m[a], label="m⁰=$(m0[a])", c=cg[a+1], m=:o, msc=:auto)
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
    # plot!(ple, 1:T, c[a], label="m⁰=$(m0[a])", c=cg[a+1], m=:o, msc=:auto)
    plot!(ple, Δt:Δt:(T*Δt), -3/2*c[a], label="m⁰=$(m0[a])", c=cg[a+1], m=:o, msc=:auto, ylabel=L"E")
end
ple

plot(plm, ple, size=(800,400))

##### david
using DelimitedFiles
names = reverse([
    "MC_Voter_cont_time_ener_N_1000_c_3_00_m0_0_70_t0_0_00_tl_50_00_ts.txt",
    "MC_Voter_cont_time_ener_N_1000_c_3_00_m0_0_80_t0_0_00_tl_50_00_ts.txt",
    "MC_Voter_cont_time_ener_N_1000_c_3_00_m0_0_90_t0_0_00_tl_50_00_ts.txt"
])
e_david = map(names) do fn
    y = readdlm(fn)
    vec(y[:,2])
end
ple2 = deepcopy(ple)
for a in eachindex(samplers)
    plot!(ple2, LinRange(0,50,length(e_david[a])), e_david[a], label="", c=cg[a+1])
end
plot(ple2, xlims=(0,10))

# using DelimitedFiles
# open("energy.txt", "w") do io
#     E = reduce(hcat, hcat(1:T, -mm*ne(g) / nv(g)) for mm in c)
#     writedlm(io, E)
# end
# open("magnetiz.txt", "w") do io
#     M = reduce(hcat, hcat(0:T, mm) for mm in m)
#     writedlm(io, M)
# end

function histogram_magnetiz(sms; t=size(sms.X[1],2))
    [mean(x[:,t] .|> spin) for x in sms.X]
end

anim = @animate for t in 1:T+1
h = plot([(histogram(histogram_magnetiz(sms; t); c=cg[a+1], msc=:auto, lw=0.5, label="$(m0[a])", 
    normalize=:probability, bins=LinRange(-1,1,50)); vline!([m0[a]], c=cg[a+1], label=""))
    for (a,sms) in enumerate(samplers)]..., layout=((length(m0)),1), size=(400,200*(length(m0))), title="t=$t",
    xlabel=L"m", xlims=(-1,1), ylims=(0,0.3), margin=5Plots.mm, legend=:topleft)
end
gif(anim; fps=10, loop=0)

########## MPBP

# sms = samplers[1]
# bp = deepcopy(sms.bp)

# cb = CB_BP(bp; showprogress=true)
# maxiter = 10
# svd_trunc = TruncBond(10)
# tol = 1e-5
# iterate!(bp; maxiter, svd_trunc, cb, tol)