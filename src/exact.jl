# first turn integer `x` into its binary representation, then reshape the
#  resulting bit vector into a matrix of size specified by `dims`
function _int_to_matrix(x::Integer, q::Integer, dims)
    y = digits(x, base=q, pad=prod(dims))
    return reshape(y, dims) .+ 1
end

function exact_prob(bp::MPBP{G,F,U}) where {G,F,U}
    @unpack g, w, ϕ, ψ = bp
    N = nv(g); T = getT(bp); q = nstates(U)
    T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
    
    logp = zeros(q^(N*(T+1)))
    prog = Progress(q^(N*(T+1)), desc="Computing joint probability")
    X = zeros(Int, T+1, N)
    for x in 1:q^(N*(T+1))
        X .= _int_to_matrix(x-1, q, (T+1,N))
        for i in 1:N
            logp[x] += log( ϕ[i][1][X[1,i]] )
            ∂i = neighbors(g, i)
            for t in 1:T
                logp[x] += log( w[i][t](X[t+1,i], X[t,∂i], X[t,i]) )
                logp[x] += log( ϕ[i][t+1][X[t+1,i]] )
            end
        end
        for (i, j, ij) in edges(g)
            for t in 1:T+1
                logp[x] += 1/2 * log( ψ[ij][t][X[t,i],X[t,j]] )
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

function site_marginals(bp::MPBP{G,F,U}; p = exact_prob(bp)[1]) where {G,F,U}
    N = nv(bp.g); T = getT(bp); q = nstates(U)
    m = [zeros(fill(2,T+1)...) for i in 1:N]
    prog = Progress(2^(N*(T+1)), desc="Computing exact marginals")
    X = zeros(Int, T+1, N)
    for x in 1:q^(N*(T+1))
        X .= _int_to_matrix(x-1, q, (T+1,N))
        for i in 1:N
            m[i][X[:,i]...] += p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    m
end

function exact_marginals(bp::MPBP{G,F,U}; p_exact = exact_prob(bp)[1]) where {G,F,U}
    m = site_marginals(bp; p = p_exact)
    N = nv(bp.g); T = getT(bp); q = nstates(U)
    pp = [[zeros(q) for t in 0:T] for i in 1:N]
    for i in 1:N
        for t in 1:T+1
            for xᵢᵗ in 1:q
                indices = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                pp[i][t][xᵢᵗ] = sum(m[i][indices...])
            end
            # pp[i][t] ./= sum(pp[i][t])
        end
    end
    pp
end

function exact_marginal_expectations(bp::MPBP{G,F,U}; 
        m_exact = exact_marginals(bp)) where {G,F,U}
    μ = [zeros(getT(bp)+1) for _ in eachindex(m_exact)]
    for i in eachindex(m_exact)
        for t in eachindex(m_exact[i])
            μ[i][t] = marginal_to_expectation(m_exact[i][t], U)
        end
    end
    μ
end

function exact_autocorrelations(bp::MPBP{G,F,U}; 
        p_exact = exact_prob(bp)[1]) where {G,F,U}
    m = site_marginals(bp; p = p_exact)
    N = nv(bp.g); T = getT(bp); q = nstates(U)
    r = [zeros(T+1, T+1) for i in 1:N]
    for i in 1:N
        for u in axes(r[i], 2), t in 1:u-1
            p = zeros(q, q)
            for xᵢᵗ in 1:q, xᵢᵘ in 1:q
                indices = map(1:T+1) do s
                    if s == t
                        return xᵢᵗ
                    elseif s == u
                        return xᵢᵘ
                    else
                        return Colon()
                    end
                end
                p[xᵢᵗ, xᵢᵘ] = sum(m[i][indices...])
            end 
            r[i][t, u] = marginal_to_expectation(p, U)
        end
    end
    r
end

function exact_autocovariances(bp::MPBP;
        r = exact_autocorrelations(bp), μ = exact_marginal_expectations(bp))
    _autocovariances(r, μ)
end

