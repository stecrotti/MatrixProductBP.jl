include("../sis.jl")
include("../mpdbp.jl")
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
function find_zero_patients_bp(bp::MPdBP;
        svd_trunc = TruncBond(5), tol=1e-3, maxiter=100) where {F<:Real}
    cb = CB_BP(bp)
    iters, ε = iterate!(bp; maxiter, svd_trunc, cb, tol)
    iters == maxiter && error("BP did not converge")
    b = beliefs(bp)
    b⁰ = [bb[1] for bb in b]
    p = sortperm(b⁰, by=x->x[1])
    eachindex(b⁰)[p], [bb[I] for bb in b⁰[p]]
end

function find_zero_patients_mc(bp::MPdBP; nsamples=10^4) where {F<:Real}
    sms = sample(bp, nsamples)
    b = marginals(sms)
    b⁰ = [bb[1] for bb in b]
    p = sortperm(b⁰, by=x->x[1])
    eachindex(b⁰)[p], [bb[I] for bb in b⁰[p]], sms
end

# compute ROC curve
function roc(guess_zp, true_zp)
    r = guess_zp .∈ (true_zp,)
    sr = sum(r)
    sr == 0 && return zeros(length(r)), ones(length(r)) 
    cumsum(.!r), cumsum(r)
end

function auc(guess_zp, true_zp)
    x, y = roc(guess_zp, true_zp)
    Z = maximum(x) * maximum(y)
    Z == 0 && return 1.0
    a = 0
    for i in 2:length(y)
        if y[i] == y[i-1]
            a += y[i]
        end  
    end
    a / Z 
end
