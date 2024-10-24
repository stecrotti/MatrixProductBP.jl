# first turn integer `x` into its binary representation, then reshape the
#  resulting bit vector into a matrix of size specified by `dims`
_int_to_matrix(x, qs, dims) = reshape(CartesianIndices(qs)[x] |> Tuple |> collect, dims)

function exact_prob(bp::MPBP{G,F}) where {G,F}
    @unpack g, w, ϕ, ψ = bp
    N = nv(g); T = getT(bp);
    T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
    
    Q = prod(nstates(bp,i)^(T+1) for i=1:N)
    qs = Tuple(nstates(bp,i) for t=1:T+1, i=1:N)
    logp = zeros(Q)
    prog = Progress(Q, desc="Computing joint probability")
    X = zeros(Int, T+1, N)
    for x in 1:Q
        X .= _int_to_matrix(x, qs, (T+1,N))
        for i in 1:N
            logp[x] += log( ϕ[i][1][X[1,i]] )
            ∂i = neighbors(g, i)
            for t in 1:T
                logp[x] += log( w[i][t](X[t+1,i], X[t,∂i], X[t,i]) )
                logp[x] += log( ϕ[i][t+1][X[t+1,i]] )
            end
            if is_periodic(bp)
                logp[x] += log( w[i][end](X[1,i], X[end,∂i], X[end,i]) )
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

function site_marginals(bp::MPBP{G,F}; p = exact_prob(bp)[1]) where {G,F}
    N = nv(bp.g); T = getT(bp);
    qs = Tuple(nstates(bp,i) for t=1:T+1, i=1:N)
    m = [zeros(fill(nstates(bp,i),T+1)...) for i in 1:N]
    prog = Progress(prod(qs), desc="Computing exact marginals")
    X = zeros(Int, T+1,N)
    for x in 1:prod(qs)
        X .= _int_to_matrix(x, qs, (T+1,N))
        for i in 1:N
            m[i][X[:,i]...] += p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    m
end

function exact_marginals(bp::MPBP{G,F}; p_exact = exact_prob(bp)[1]) where {G,F}
    m = site_marginals(bp; p = p_exact)
    N = nv(bp.g); T = getT(bp)
    pp = [[zeros(nstates(bp,i)) for t in 0:T] for i in 1:N]
    for i in 1:N
        for t in 1:T+1
            for xᵢᵗ in 1:nstates(bp,i)
                indices = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                pp[i][t][xᵢᵗ] = sum(m[i][indices...])
            end
            # pp[i][t] ./= sum(pp[i][t])
        end
    end
    pp
end

function exact_marginal_expectations(f, bp::MPBP{G,F}; 
        m_exact = exact_marginals(bp)) where {G,F}
    map(eachindex(m_exact)) do i
        expectation.(x->f(x,i), m_exact[i])
    end
end

exact_marginal_expectations(bp; m_exact = exact_marginals(bp)) = exact_marginal_expectations((x,i)->x, bp; m_exact)

function pair_marginals(bp::MPBP{G,F}; p = exact_prob(bp)[1]) where {G,F}
    T = getT(bp); N = nv(bp.g)
    qs = Tuple(nstates(bp,i) for t=1:T+1, i=1:N)
    m = [zeros(vcat(fill(nstates(bp,i),T+1), fill(nstates(bp,j),T+1))...) for (i,j) in edges(bp.g)]
    prog = Progress(prod(qs), desc="Computing exact pair marginals")
    X = zeros(Int, T+1, N)
    for x in 1:prod(qs)
        X .= _int_to_matrix(x, qs, (T+1,N))
        for (i,j,id) in edges(bp.g)
            m[id][X[:,i]...,X[:,j]...] += p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    return m
end

function exact_pair_marginals(bp::MPBP{G,F}; p_exact = exact_prob(bp)[1]) where {G,F}
    m = pair_marginals(bp; p = p_exact)
    T = getT(bp)
    pp = [[zeros(nstates(bp,i),nstates(bp,j)) for t in 1:T+1] for (i,j) in edges(bp.g)]
    for (i,j,id) in edges(bp.g)
        for t in 1:T+1
            for xᵢᵗ in 1:nstates(bp,i)
                for xⱼᵗ⁺¹ in 1:nstates(bp,j)
                    indices_i = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                    indices_j = [s==t ? xⱼᵗ⁺¹ : Colon() for s in 1:T+1]
                    pp[id][t][xᵢᵗ,xⱼᵗ⁺¹] += sum(m[id][indices_i...,indices_j...])
                end
            end
            @debug @assert sum(pp[id][t]) ≈ 1
        end
    end
    return pp
end

function exact_pair_marginal_expectations(f, bp::MPBP{G,F}; 
        m_exact = exact_pair_marginals(bp)) where {G,F}
    map(eachindex(m_exact)) do i
        expectation.(x->f(x,i), m_exact[i])
    end
end

function exact_pair_marginal_expectations(bp; m_exact = exact_alternate_marginals(bp)) 
    return exact_pair_marginal_expectations((x,i)->x, bp; m_exact)
end

function exact_alternate_marginals(bp::MPBP{G,F}; p_exact = exact_prob(bp)[1]) where {G,F}
    m = pair_marginals(bp; p = p_exact)
    T = getT(bp)
    pp = [[zeros(nstates(bp,i),nstates(bp,j)) for t in 1:T] for (i,j) in edges(bp.g)]
    for (i,j,id) in edges(bp.g)
        for t in 1:T
            for xᵢᵗ in 1:nstates(bp,i)
                for xⱼᵗ⁺¹ in 1:nstates(bp,j)
                    indices_i = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                    indices_j = [s==t+1 ? xⱼᵗ⁺¹ : Colon() for s in 1:T+1]
                    pp[id][t][xᵢᵗ,xⱼᵗ⁺¹] = sum(m[id][indices_i...,indices_j...])
                end
            end
        end
    end
    return pp
end

function exact_alternate_marginal_expectations(f, bp::MPBP{G,F}; 
        m_exact = exact_alternate_marginals(bp)) where {G,F}
    map(eachindex(m_exact)) do i
        expectation.(x->f(x,i), m_exact[i])
    end
end
function exact_alternate_marginal_expectations(bp; m_exact = exact_alternate_marginals(bp)) 
    return exact_alternate_marginal_expectations((x,i)->x, bp; m_exact)
end


function exact_autocorrelations(f, bp::MPBP{G,F}; 
        p_exact = exact_prob(bp)[1]) where {G,F}
    m = site_marginals(bp; p = p_exact)
    N = nv(bp.g); T = getT(bp);
    r = [zeros(T+1, T+1) for i in 1:N]
    for i in 1:N
        for u in axes(r[i], 2), t in 1:u-1
            qi = nstates(bp,i)
            p = zeros(qi, qi)
            for xᵢᵗ in 1:qi, xᵢᵘ in 1:qi
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
            r[i][t, u] = expectation(x->f(x,i), p)
        end
    end
    r
end

exact_autocorrelations(bp::MPBP; p_exact = exact_prob(bp)[1]) = exact_autocorrelations((x,i)->x, bp; p_exact)


function exact_autocovariances(f, bp::MPBP;
        r = exact_autocorrelations(f, bp), μ = exact_marginal_expectations(f, bp))
    covariance.(r, μ)
end

function exact_autocovariances(bp::MPBP; r = exact_autocorrelations(bp), μ = exact_marginal_expectations(bp))
    covariance.(r, μ)
end