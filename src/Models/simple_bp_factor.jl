# for a `SimpleBPFactor`, outgoing messages can be computed recursively
abstract type SimpleBPFactor <: BPFactor; end

# number of states for variable which accumulates the first `l` neighbors
nstates(::Type{<:SimpleBPFactor}, l::Integer) = error("Not implemented")

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial(mₗᵢ::MPEM2, mᵢⱼₗ₁::MPEM2, 
        wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {U<:SimpleBPFactor}
    T = getT(mₗᵢ)
    @assert getT(mᵢⱼₗ₁) == T
    map(1:T+1) do t
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        qxᵢ = nstates(U); qy = nstates(U, l)
        AAᵗ = zeros(size(Aᵗ, 1), size(Aᵗ, 2), qy, qxᵢ)
        if t ≤ T
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg(wᵢ[min(T, $t)],yₗᵗ,yₗ₁ᵗ,xₗᵗ,l) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        else
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = 1/qy * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        end
    end |> MPEM2
end


# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
    d::Integer; prob = prob_ijy) where {U<:SimpleBPFactor}
    q = nstates(U)
    B = [zeros(q, q, size(a,1), size(a,2), q) for a in A]
    for t in 1:getT(A)
        Aᵗ,Bᵗ = A[t], B[t]
        @tullio Bᵗ[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[xᵢᵀ,xⱼᵀ,m,n,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && println("NaN in tensor train")
    return MPEM3(B)
end

function f_bp(A::Vector{MPEM2{F}},
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}},
        j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}

    d = length(A) - 1   # number of neighbors other than j
    @assert j ∈ eachindex(A)
    T = getT(A[1])
    @assert all(getT(a) == T for a in A)

    # initialize recursion
    qxᵢ = nstates(U); qy = nstates(U, 0)
    M = reshape(vcat(ones(1,qxᵢ), zeros(qy-1,qxᵢ)), (1,1,qy,qxᵢ))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    logz = 0.0
    for (l,k) in enumerate(k for k in eachindex(A) if k != j)
        mᵢⱼₗ₁ = f_bp_partial(A[k], mᵢⱼₗ₁, wᵢ, ψₙᵢ[k], l)
        logz +=  normalize!(mᵢⱼₗ₁)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, wᵢ, ϕᵢ, d; prob = prob_ijy)

    return B, logz
end

function f_bp_dummy_neighbor(A::Vector{MPEM2{F}}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        svd_trunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}
    
    d = length(A)
    T = getT(A[1]); q = nstates(U)
    @assert all(getT(a) == T for a in A)

    # initialize recursion
    qxᵢ = nstates(U); qy = nstates(U, 0)
    M = fill(1.0, 1, 1, 1, qxᵢ)
    #M = reshape(vcat(ones(1,qxᵢ), zeros(qy-1,qxᵢ)), (1,1,qy,qxᵢ))
    mᵢⱼₗ₁ = MPEM2(fill(M, T+1))

    logz = 0.0
    # compute partial messages from all neighbors
    for l in eachindex(A)
        mᵢⱼₗ₁ = f_bp_partial(A[l], mᵢⱼₗ₁, wᵢ, ψₙᵢ[l], l)
        logz += normalize!(mᵢⱼₗ₁)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, wᵢ, ϕᵢ, d; prob = prob_ijy_dummy)

    return B, logz
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{F,U}, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F,U}
    @unpack g, w, ϕ, ψ, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]; ψout = ψ[eout.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    logzᵢ = 0.0
    B = map(eachindex(A)) do k
        Bk = map(1:getT(A[k])) do t
            Ak = A[k][t]
            qy = nstates(U, 1)
            wᵢᵗ = bp.w[i][t]
            ψᵢₖᵗ = ψout[k][t]
            Bkt = zeros(size(Ak,1), size(Ak,2), qy, size(Ak,4))
            @tullio Bkt[m,n,yₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ) * Ak[m,n,xₖ,xᵢ] * ψᵢₖᵗ[xₖ,xᵢ]
        end |> MPEM2
        Bk, 1, 0.0
    end

    M = fill(1.0, 1, 1, 1, qxᵢ)
    init = (MPEM2(fill(M, T+1)), 0, 0.0)


    function op((B1, n1 ,lz1), (B2, n2, lz2))
        B = map(eachindex(B1.tensors)) do t
            Bout = zeros(size(B1[t],1), size(B1[t],2), nstates(U,n1+n2), size(B1[t],4))
            Bᵗ = kron2(B1[t], B2[t])
            @tullio Bout[m,n,y,xᵢ] = prob_yy(bp.w[$i][$t],y,y1,y2,xᵢ) * Bᵗ[m,n,xᵢ,y1,y2]
        end |> MPEM2
        logz = normalize!(B)
        # SVD L to R with no truncation
        sweep_LtoR!(B, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(B; svd_trunc)
        B, n1 + n2, logz + lz1 + lz2
    end

    dest, (full,_,logzi)  = cavity(B, op, init)
    (C,_,logzs) = unzip(dest)

    for k in eachindex(A)
        for t in eachindex()
    end

    for (j_ind, e_out) in enumerate(eout)
        B, logzᵢ₂ⱼ = f_bp(A, w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        logzᵢ₂ⱼ += normalize!(μ[idx(e_out)])
        logzᵢ += logzᵢ₂ⱼ
    end
    dᵢ = length(ein)
    bp.b[i] = onebpiter_dummy_neighbor(bp, i; svd_trunc) |> marginalize
    return (1 / dᵢ) * logzᵢ
end

function beliefs(bp::MPBP{F,U};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}
    [marginals(bi) for bi in bp.b]
end

function beliefs_tu(bp::MPBP{F,U};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}
    q = nstates(U); T = getT(bp)
    b = [[zeros(q, q) for _ in 0:T, _ in 0:T] for _ in vertices(bp.g)]
    for i in eachindex(b)
        A = onebpiter_dummy_neighbor(bp, i; svd_trunc)
        b[i] .= firstvar_marginal_tu(A)
    end
    b
end

### INFINITE REGULAR GRAPHS

function onebpiter_infinite_graph(A::MPEM2, k::Integer, wᵢ::Vector{U}, 
        ϕᵢ, ψₙᵢ;
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:SimpleBPFactor}

    B, _ = f_bp(fill(A, k), wᵢ, ϕᵢ, ψₙᵢ, 1)
    C = mpem2(B)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    A_new
end

function iterate_bp_infinite_graph(T::Integer, k::Integer, wᵢ::Vector{U},
        ϕᵢ = fill(ones(nstates(U)), T+1);
        ψₙᵢ = fill(fill(ones(nstates(U), nstates(U)), T+1), k),
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true) where {U<:SimpleBPFactor}
    @assert length(ϕᵢ) == T + 1
    @assert length(wᵢ) == T
    
    A = mpem2(nstates(U), T)
    Δs = fill(NaN, maxiter)
    m = firstvar_marginal(A)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(maxiter; dt, desc="Iterating BP on infinite graph")
    for it in 1:maxiter
        A = onebpiter_infinite_graph(A, k, wᵢ, ϕᵢ, ψₙᵢ; svd_trunc)
        m_new = firstvar_marginal(A)
        Δ = maximum(abs, bb_new[1] - bb[1] for (bb_new, bb) in zip(m_new, m))
        Δs[it] = Δ
        Δ < tol && return A, it, Δs
        m, m_new = m_new, m
        rounded_Δ = round(Δ, digits=ceil(Int,abs(log(tol))))
        next!(prog, showvalues=[(:iter, "$it/$maxiter"), (:Δ,"$rounded_Δ/$tol")])
    end
    A, maxiter, Δs
end

function onebpiter_dummy_infinite_graph(A::MPEM2, k::Integer,
        wᵢ::Vector{U}, ϕᵢ, ψₙᵢ; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:SimpleBPFactor}

    B, _ = f_bp_dummy_neighbor(fill(A, k), wᵢ, ϕᵢ, ψₙᵢ)
    C = mpem2(B)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    A_new
end

# A is the message already converged
# return marginals, expectations of marginals and covariances
function observables_infinite_graph(A::MPEM2, k::Integer, 
        wᵢ::Vector{<:U}, ϕᵢ;
        ψₙᵢ = fill(fill(ones(nstates(U),nstates(U)), length(A)), k),
        svd_trunc::SVDTrunc=TruncThresh(1e-6), 
        showprogress=true) where {U<:SimpleBPFactor}

    Anew = onebpiter_dummy_infinite_graph(A, k, wᵢ, ϕᵢ, ψₙᵢ; svd_trunc)
    b = firstvar_marginal(Anew)
    b_tu = firstvar_marginal_tu(Anew; showprogress)
    r = marginal_to_expectation.(b_tu, (U,))
    m = marginal_to_expectation.(b, (U,))
    c = MatrixProductBP.covariance(r, m)
    b, m, c
end