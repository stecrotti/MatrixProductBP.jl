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
        @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg(wᵢ[$t],yₗᵗ,yₗ₁ᵗ,xₗᵗ,l) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
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
T = length(wᵢ)-1; q = nstates(U)
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
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F<:Real,U<:SimpleBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    dᵢ = length(ein)
    wᵢ, ϕᵢ = w[i], ϕ[i]
    T = getT(bp)
    A, ψout = μ[ein.|>idx], ψ[eout.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
 
    B = map(1:dᵢ) do k
        Bk = map(1:getT(A[k]) + 1) do t
            Akt, wᵢᵗ, ψᵢₖᵗ = A[k][t], wᵢ[t], ψout[k][t]
            Bkt = zeros(size(Akt,1), size(Akt,2), nstates(U, 1), size(Akt,4))
            @tullio Bkt[m,n,yₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ) * Akt[m,n,xₖ,xᵢ] * ψᵢₖᵗ[xᵢ,xₖ]
        end |> MPEM2
        Bk, 0.0, 1
    end

    Minit = fill(1.0, 1, 1, 1, nstates(U))
    init = (MPEM2(fill(Minit, T + 1)), 0.0, 0)

    function op((B1, lz1, n1), (B2, lz2, n2))
        B = map(eachindex(B1.tensors)) do t
            Bᵗ = kron2(B1[t], B2[t])
            Bout = zeros(size(Bᵗ,1), size(Bᵗ,2), nstates(U,n1+n2), size(Bᵗ,3))
            @tullio Bout[m,n,y,xᵢ] = prob_yy(wᵢ[$t],y,y1,y2,xᵢ) * Bᵗ[m,n,xᵢ,y1,y2]
        end |> MPEM2
        lz = normalize!(B)
        sweep_LtoR!(B, svd_trunc=TruncThresh(0.0))
        sweep_RtoL!(B; svd_trunc)
        B, lz + lz1 + lz2, n1 + n2
    end

    dest, (full, logzᵢ2)  = cavity(B, op, init)
    (C, logzs) = unzip(dest)

    logzᵢ = sum(logzs)
    for (j,e) = enumerate(eout)
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ - 1; prob = prob_ijy)
        μ[idx(e)] = sweep_RtoL!(mpem2(B); svd_trunc)
        logzᵢ += normalize!(μ[idx(e)])
    end
    B = f_bp_partial_ij(full, wᵢ, ϕᵢ, dᵢ; prob = prob_ijy_dummy)
    bp.b[i] = B |> mpem2 |> marginalize
    return dᵢ == 0 ? 0.0 : logzᵢ / dᵢ
end


function beliefs(bp::MPBP{F,U};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}
    [marginals(bi) for bi in bp.b]
end

function beliefs_tu(bp::MPBP{F,U};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {F,U<:SimpleBPFactor}
    [marginals_tu(bi) for bi in bp.b]
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