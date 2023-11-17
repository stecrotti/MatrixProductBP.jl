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

struct ExactMsg{F<:Real,S,TI<:Integer}
    logm       :: Vector{F}
    states  :: S
    T       :: TI

    function ExactMsg(logm::Vector{F}, states::S, T::TI) where {F<:Real,S,TI<:Integer}
        @assert length(logm) == prod(states) ^ (T+1)
        new{F,S,TI}(logm, states, T)
    end
end

function uniform_exact_msg(states, T)
    n = prod(states) ^ (T+1)
    x = log(1 / n)
    m = fill(x, n)
    return ExactMsg(m, states, T)
end

nstates(m::ExactMsg) = prod(m.states)
Base.length(m::ExactMsg) = m.T + 1
function normalize!(m::ExactMsg) 
    logz = logsumexp(m.logm)
    m.logm .-= logz
    return logz
end
normalization(m::ExactMsg) = exp(logsumexp(m.logm))

function eachstate(m::ExactMsg, args...)
    return zip(eachindex(m.logm), Iterators.product(fill(Iterators.product([1:s for s in m.states]...), m.T+1)...))
end

const MPBPExact = MPBP{<:AbstractIndexedDiGraph, <:Real, <:AbstractVector{<:BPFactor}, <:ExactMsg, <:ExactMsg}

function mpbp_exact(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}},
        q::AbstractVector{Int}, T::Int; 
        ϕ = [[ones(q[i]) for t in 0:T] for i in vertices(g)],
        ψ = [[ones(q[i],q[j]) for t in 0:T] for (i,j) in edges(g)],
        μ = [uniform_exact_msg((q[i],q[j]), T) for (i,j) in edges(g)],
        b = [uniform_exact_msg((q[i],), T) for i in vertices(g)],
        f = zeros(nv(g)))
    return MPBP(g, w, ϕ, ψ, μ, b, f)
end

function f_bp(m_in::Vector{M2}, wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, 
        ψₙᵢ::Vector{Vector{Matrix{F}}}, j_index::Integer; showprogress=false, 
        periodic=false) where {F,U<:BPFactor,M2<:ExactMsg}
    T = length(m_in[1]) - 1
    @assert all(length(a) == T + 1 for a in m_in)
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1

    dt = showprogress ? 1.0 : Inf
    prog = Progress(prod(nstates, m_in), dt=dt, desc="Computing outgoing message")
    # m_in[j_index].logm .= -Inf
    m_out = deepcopy(m_in[j_index])
    m_out.logm .= -Inf
    for xₐ in Iterators.product((eachstate(m) for m in m_in)...)
        # compute weight
        y = prod(1:(T+1)) do t
            xᵢᵗ⁺¹ = xₐ[j_index][2][mod1(t+1,T+1)][1]
            xₙᵢᵗ = [xₐ[k][2][mod1(t,T+1)][2] for k in eachindex(xₐ) if k != j_index]
            xᵢᵗ = xₐ[j_index][2][mod1(t,T+1)][1]
            w = (t!=T+1 || periodic) ? wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ) : 1
            ϕ = ϕᵢ[t][xᵢᵗ]
            ψ = prod(ψₖᵢ[t][xₖᵗ,xᵢᵗ] for (xₖᵗ, ψₖᵢ) in zip(xₙᵢᵗ,ψₙᵢ); init=1)
            w * ϕ * ψ
        end
        # compute (log of) product of incoming messages
        z = sum(m_in[k].logm[xₖᵢ[1]] for (xₖᵢ,k) in zip(xₐ, eachindex(xₐ)) if k != j_index; init=0)
        # sum
        m_out.logm[xₐ[j_index][1]] = logaddexp(
            m_out.logm[xₐ[j_index][1]], log(y) + z)
        next!(prog)
    end
    return m_out
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBPExact, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6), damp=0.0, periodic=false) where {U<:BPFactor}
    (; g, w, ϕ, ψ, μ) = bp
    ein = inedges(g, i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    # @assert all(normalization(a) ≈ 1 for a in A)
    sumlogzᵢ₂ⱼ = 0.0
    for (j_ind, e_out) in enumerate(eout)
        μj = f_bp(A, w[i], ϕ[i], ψ[eout.|>idx], j_ind; periodic)
        sumlogzᵢ₂ⱼ += normalize!(μj)
        μ[idx(e_out)] = μj # damp?
    end
    dᵢ = length(ein)
    bp.b[i] = onebpiter_dummy_neighbor(bp, i; svd_trunc) |> marginalize
    logzᵢ = log(normalization(bp.b[i]))
    bp.f[i] = (dᵢ/2-1)*logzᵢ - (1/2)*sumlogzᵢ₂ⱼ
    nothing
end