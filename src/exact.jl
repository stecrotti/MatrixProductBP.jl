# first turn integer `x` into its binary representation, then reshape the
#  resulting bit vector into a matrix of size specified by `dims`
function _int_to_matrix(x::Integer, dims)
    y = digits(x, base=2, pad=prod(dims))
    return reshape(y, dims) .+ 1
end

function exact_prob(bp::MPBP{q,T,F,U}) where {q,T,F,U}
    @assert q==2 "Can compute exact prob only for binary variables (for now)"
    @unpack g, w, p⁰, ϕ, ψ = bp
    N = nv(g)
    T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
    
    logp = zeros(2^(N*(T+1)))
    prog = Progress(2^(N*(T+1)), desc="Computing joint probability")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= _int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            logp[x] += log( p⁰[i][X[1,i]] )
            ∂i = neighbors(g, i)
            for t in 1:T
                logp[x] += log( w[i][t](X[t+1,i], X[t,∂i], X[t,i]) )
                logp[x] += log( ϕ[i][t][X[t+1,i]] )
            end
        end
        for (i, j, ij) in edges(g)
            for t in 1:T
                logp[x] += 1/2 * log( ψ[ij][t][X[t+1,i],X[t+1,j]] )
            end
        end
        next!(prog)
    end
    logZ = logsumexp(logp)
    logp .-= logZ
    # overwrite `logp` with `p` to avoid allocating a new large array
    map!(exp, logp, logp); p = logp
    Z = exp(logZ)
    p, Z
end

function site_marginals(bp::MPBP{q,T,F,U}; 
        p = exact_prob(bp)[1],
        m = [zeros(fill(2,T+1)...) for i in 1:nv(bp.g)]) where {q,T,F,U}
    N = nv(bp.g)
    prog = Progress(2^(N*(T+1)), desc="Computing site marginals")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= _int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            m[i][X[:,i]...] += p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    m
end

function site_time_marginals(bp::MPBP{q,T,F,U}; 
        m = site_marginals(bp)) where {q,T,F,U}
    N = nv(bp.g)
    pp = [[zeros(2) for t in 0:T] for i in 1:N]
    for i in 1:N
        for t in 1:T+1
            for xᵢᵗ in 1:2
                indices = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                pp[i][t][xᵢᵗ] = sum(m[i][indices...])
            end
            pp[i][t] ./= sum(pp[i][t])
        end
    end
    pp
end