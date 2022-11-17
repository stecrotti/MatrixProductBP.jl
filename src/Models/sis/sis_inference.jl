# return individuals ranked by posterior prob at time zero
# tuple of indices and probabilities
function find_infected_bp(bp::MPBP; showprogress=false,
        svd_trunc = TruncBond(5), tol=1e-3, maxiter=100,
        require_convergence::Bool=true)

    cb = CB_BP(bp; showprogress)
    reset_messages!(bp)
    iters, _ = iterate!(bp; maxiter, svd_trunc, cb, tol)
    if require_convergence
        iters == maxiter && error("BP did not converge")
    end
    b = beliefs(bp)
    guesses = map(eachindex(b[1])) do t
        bb = [bbb[t] for bbb in b]
        p = sortperm(bb, by=x->x[1])
        eachindex(bb)[p]
    end

    guesses
end


# compute ROC curve
function roc(guess_zp::Vector{Int}, true_zp::Vector{Int})

    r = guess_zp .∈ (true_zp,)
    sr = sum(r)
    sr == 0 && return zeros(length(r)), ones(length(r)) 

    cumsum(.!r), cumsum(r)
end

function auc(guess_zp::Vector{Int}, true_zp::Vector{Int})

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


function kl(q::Vector{<:Real}, p::Vector{<:Real})

    @assert length(q)==length(p)
    k = 0.0
    for (qi,pi) in zip(p,q)
        k+= xlogx(qi) - xlogy(qi, pi)
    end

    k
end

# compute KL between guess and true marginals at all times
function kl_marginals(b_guess::U, b_true::U) where {U<:Vector{Vector{Vector{Float64}}}}
    
    N = length(b_guess); T = length(b_guess[1])-1
    @assert length(b_true) == N
    @assert all(length(bg) == length(bt) == T+1 for (bg,bt) in zip(b_guess,b_true))

    map(1:T+1) do t 
        map(1:N) do i
            kl(b_guess[i][t], b_true[i][t])
        end |> mean
    end
end

# compute L1 error between guess and true marginals at all times
function l1_marginals(b_guess::U, b_true::U) where {U<:Vector{Vector{Vector{Float64}}}}
    
    N = length(b_guess); T = length(b_guess[1])-1
    @assert length(b_true) == N
    @assert all(length(bg) == length(bt) == T+1 for (bg,bt) in zip(b_guess,b_true))

    map(1:T+1) do t 
        map(1:N) do i
            abs(b_guess[i][t][INFECTED] - b_true[i][t][INFECTED])
        end |> mean
    end
end

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
function find_zero_patients_bp(bp::MPBP; showprogress=false,
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

function find_zero_patients_mc(bp::MPBP; nsamples=10^4, 
    sms = sample(bp, nsamples))

    b = marginals(sms)
    b⁰ = [bb[1] for bb in b]
    p = sortperm(b⁰, by=x->x[1])

    eachindex(b⁰)[p], [bb[INFECTED] for bb in b⁰[p]], sms
end