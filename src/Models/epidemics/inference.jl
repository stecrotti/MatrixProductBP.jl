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
    for i in Iterators.drop(eachindex(y), 1)
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
            abs(b_guess[i][t][INFECTIOUS] - b_true[i][t][INFECTIOUS])
        end |> mean
    end
end