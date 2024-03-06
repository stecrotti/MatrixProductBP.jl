using MatrixProductBP
using MatrixProductBP.Models
using Graphs, IndexedGraphs, Random
using Plots, LaTeXStrings
using Statistics
using JLD2
using TensorTrains
using Unzip, ProgressMeter

Plots.default(
    grid = :off, box = :on,
    widen = true,
    label = "",
    lw = 1.5,
    msc = :auto,
    size = (400,300),
    markersize = 3,
    margin=5Plots.mm
)

function infer_voter(bp, p0; 
        rng = Random.default_rng(), svd_trunc = TruncThresh(5e-3),
        softinf=Inf)
    N = nv(bp.g)
    reset!(bp, observations=true)
    for i in 1:N
        bp.ϕ[i][1] .= [p0, 1-p0]
    end
    X = draw_node_observations!(bp, N; rng, last_time=true, softinf)
    svd_trunc = TruncThresh(5e-3)
    cb = CB_BP(bp; showprogress=false)
    iterate!(bp; maxiter=20, svd_trunc, cb, tol=1e-6, damp=0.6)
    bp, X, cb
end

rng = MersenneTwister(111)

T = 12
N = 100
p0 = 0.7
gg = prufer_decode(rand(rng, 1:N, N-2))
J = 1
w = fill(fill(HomogeneousVoterFactor(J), T+1), N)

bp = mpbp(IndexedBiDiGraph(gg), w, fill(2,N), T);
svd_trunc = TruncThresh(5e-3)

nsamples = 10
bps, Xs, cbs = map(1:nsamples) do s
    bp_, X, cb = infer_voter(deepcopy(bp), p0; rng, svd_trunc)
    println("Finished sample $s of $(nsamples)")
    bp_, X, cb
end |> unzip;

Δts = 0:T

props = map(zip(bps,Xs)) do (bp,X)
    b_bp = beliefs(bp)
    prop = map(Δts) do Δt
        x_t = X[1][:,end-Δt]
        p_bp_t = [p[end-Δt] for p in b_bp]        
        mean(argmax(p)==x for (p,x) in zip(p_bp_t, x_t))
    end
end;

props_naive = map(zip(bps,Xs)) do (bp,X)
    b_bp = beliefs(bp)
    prop = map(Δts) do Δt
        x_t = X[1][:,end-Δt]
        y = X[1][:,end]
        mean(yy==x for (yy,x) in zip(y, x_t))
    end
end;

maxbonddim(bp::MPBP) = maximum(maximum.(bond_dims.(bp.μ)))
av_bp = round(Int, mean(maxbonddim.(bps)))

pl = plot(Δts, mean(props), yerr=std(props)./sqrt(nsamples),
    m=:o, xlabel=L"\Delta t", size=(500,400), label="MPBP",
    ylabel=L"\frac{1}{N}\sum_{i=1}^N \delta(X_i^{T-\Delta t}, \arg\max p_i^{T-\Delta t})",
    title="Prediction Δt steps in the past\n Tree graph, N=$N, T=$T, $nsamples samples\n Avg matrix size $av_bp")

plot!(pl, Δts, mean(props_naive), yerr=std(props)./sqrt(nsamples),
    m=:o, label="naive")
plot!(pl, titlefontsize=9, margin=15Plots.mm, size=(700,400))
savefig(pl, (@__DIR__)*"/voter_prediction_short_time.pdf")

include("../../telegram/notifications.jl")
@telegram "voter bp"
