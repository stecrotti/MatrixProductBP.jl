import StatsBase: sample
import LogExpFunctions: logistic, softmax!
import Graphs: random_regular_graph

include("inference.jl")
include("../sis.jl")
include("../exact/montecarlo.jl")

# return epidemic instance as an `N`-by-`T`+1 matrix and `MPdBP` object
function simulate_sis(g::IndexedGraph, λ::Real, κ::Real, p⁰::Vector{Vector{F}}, T::Integer, 
        nobs::Integer; softinf=1e3) where {F<:Real}

    sis = SIS(g, λ, κ, T; p⁰)
    bp = mpdbp(sis)
    X, _ = onesample(bp)
    draw_node_observations!(bp.ϕ, X, nobs; softinf, last_time=true)

    X, bp
end

# return individuals ranked by posterior prob at time zero
# tuple of indices and probabilities
function find_zero_patients_bp(bp::MPdBP; showprogress=false,
        svd_trunc = TruncBond(5), tol=1e-3, maxiter=100,
        require_convergence::Bool=true)
    cb = CB_BP(bp; showprogress)
    reset_messages!(bp)
    iters, ε = iterate!(bp; maxiter, svd_trunc, cb, tol)
    if require_convergence
        iters == maxiter && error("BP did not converge")
    end
    b = beliefs(bp)
    b⁰ = [bb[1] for bb in b]
    p = sortperm(b⁰, by=x->x[1])
    eachindex(b⁰)[p], [bb[I] for bb in b⁰[p]]
end

function find_zero_patients_mc(bp::MPdBP; nsamples=10^4, 
        sms = sample(bp, nsamples))
    
    b = marginals(sms)
    b⁰ = [bb[1] for bb in b]
    p = sortperm(b⁰, by=x->x[1])
    eachindex(b⁰)[p], [bb[INFECTED] for bb in b⁰[p]], sms
end


function softmax(β::Real, N::Integer, i::Integer)
    @assert i ∈ 1:N
    β = float(β)
    softmax!( [j==i ? β : -β for j in 1:N] )
end
