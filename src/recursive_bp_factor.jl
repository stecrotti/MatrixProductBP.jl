""""
For a `w::U` where `U<:RecursiveBPFactor`, outgoing messages can be computed recursively
A `<:RecursiveBPFactor` must implement: `nstates`, `prob_y`, `prob_xy` and `prob_yy`
Optionally, it can also implement `prob_y_partial` and `(w::U)(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)`
"""
abstract type RecursiveBPFactor <: BPFactor; end

#### the next five methods are the minimal needed interface for a new <:RecursiveBPFactor

"Number of states for aux variable which accumulates the first `l` neighbors"
nstates(::Type{<:RecursiveBPFactor}, l::Integer) = error("Not implemented")

"P(xᵢᵗ⁺¹|xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ)
Might depend on the degree `dᵢ` because of a change of variable from 
    y ∈ {1,2,...} to its physical value, e.g. {-dᵢ,...,dᵢ} for Ising"
prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U<:RecursiveBPFactor} = error("Not implemented")

"P(yₖᵗ| xₖᵗ, xᵢᵗ)"
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ) = error("Not implemented")
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ, dᵢ) = prob_xy(wᵢ, yₖ, xₖ, xᵢ)

"P(yₐᵦ|yₐ,yᵦ,xᵢᵗ)"
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ, d1, d2) = prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ)
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ) = error("Not implemented")


##############################################
#### the next methods are optional


"P(xᵢᵗ⁺¹|xᵢᵗ, xₙᵢᵗ, d)"
function (wᵢ::RecursiveBPFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
    xᵢᵗ::Integer)
    U = typeof(wᵢ)
    d = length(xₙᵢᵗ)
    Pyy = fill(1.0, 1)
    for k in 1:d
        Pyy = [sum(prob_yy(wᵢ, y, y1, y2, xᵢᵗ, 1, k-1)*prob_xy(wᵢ, y1, xₙᵢᵗ[k], xᵢᵗ,k)*Pyy[y2]
                   for y1 in 1:nstates(U,1), y2 in 1:nstates(U,k-1)) 
               for y in 1:nstates(U,k)]
    end
    sum(Pyy[y] * prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, y, d) for y in eachindex(Pyy))
end

"P(xᵢᵗ⁺¹|xᵢᵗ, xₙᵢᵗ, d)"
function prob_y_partial(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, k) where {U<:RecursiveBPFactor}
    sum(prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d + 1) * 
        prob_xy(wᵢ, y2, xₖᵗ, xᵢᵗ,k) * 
        prob_yy(wᵢ, yᵗ, y1, y2, xᵢᵗ, d, 1) 
        for yᵗ in 1:nstates(U, d + 1), y2 in 1:nstates(U,1))
end


#####################################################

prob_y_dummy(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, j) where U = prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, y1, d) 

# compute matrix B for mᵢⱼ
function f_bp_partial_ij(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer, qj, j) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_partial, qj, j)
end

# compute matrix B for bᵢ
function f_bp_partial_i(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_dummy, 1, 1)
end

function _f_bp_partial(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    B = [zeros(size(a,1), size(a,2), q, qj, q) for a in A]
    for t in Iterators.take(eachindex(A), length(A)-1)
        Aᵗ,Bᵗ = A[t], B[t]
        W = zeros(q,q,qj,size(Aᵗ,3))
        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j)*ϕᵢ[$t][xᵢᵗ]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ]*Aᵗ[m,n,yᵗ,xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && @error "NaN in tensor train"
    return MPEM3(B)
end

function _f_bp_partial(A::PeriodicMPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    B = [zeros(size(a,1), size(a,2), q, qj, q) for a in A]
    for t in eachindex(A)
        Aᵗ,Bᵗ = A[t], B[t]
        W = zeros(q,q,qj,size(Aᵗ,3))
        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j) * ϕᵢ[$t][xᵢᵗ]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * Aᵗ[m,n,yᵗ,xᵢᵗ]
    end
    any(any(isnan, b) for b in B) && @error "NaN in tensor train"
    return PeriodicMPEM3(B)
end

# compute ̃m{∂i∖j→i}(̅y_{∂i∖j},̅xᵢ)
function compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{M2}, ψout, T, svd_trunc) where {U<:RecursiveBPFactor, M2<:AbstractMPEM2}
    @debug (@assert all(normalization(a) ≈ 1 for a in μin))
    B = map(eachindex(ψout)) do k
        Bk = map(zip(wᵢ, μin[k], ψout[k])) do (wᵢᵗ, μₖᵢᵗ, ψᵢₖᵗ)
            Pxy = zeros(nstates(U,1), size(μₖᵢᵗ, 3), qi)
            @tullio avx=false Pxy[yₖ,xₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ,k) * ψᵢₖᵗ[xᵢ,xₖ]
            @tullio _[m,n,yₖ,xᵢ] := Pxy[yₖ,xₖ,xᵢ] * μₖᵢᵗ[m,n,xₖ,xᵢ] 
        end |> M2
        Bk, 0.0, 1
    end

    function op((B1, lz1, d1), (B2, lz2, d2))
        B = map(zip(wᵢ,B1,B2)) do (wᵢᵗ,B₁ᵗ,B₂ᵗ)
            Pyy = zeros(nstates(U,d1+d2), size(B₁ᵗ,3), size(B₂ᵗ,3), size(B₁ᵗ,4))
            @tullio avx=false Pyy[y,y1,y2,xᵢ] = prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2) 
            @tullio B3[m1,m2,n1,n2,y,xᵢ] := Pyy[y,y1,y2,xᵢ] * B₁ᵗ[m1,n1,y1,xᵢ] * B₂ᵗ[m2,n2,y2,xᵢ]
            @cast _[(m1,m2),(n1,n2),y,xᵢ] := B3[m1,m2,n1,n2,y,xᵢ]
        end |> M2
        lz = normalize!(B)
        compress!(B; svd_trunc)
        B, lz + lz1 + lz2, d1 + d2
    end

    Minit = fill(1.0, 1, 1, 1, qi)
    init = (M2(fill(Minit, T + 1)), 0.0, 0)
    dest, (full, logzᵢ,)  = cavity(B, op, init)
    (C, logzs) = unzip(dest)
    sumlogzᵢ₂ⱼ = sum(logzs)
    C, full, logzᵢ, sumlogzᵢ₂ⱼ
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{G,F}, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6), damp::Real=0.0) where {G<:AbstractIndexedDiGraph,F<:Real,U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    @assert wᵢ[1] isa U
    C, full, logzᵢ, sumlogzᵢ₂ⱼ = compute_prob_ys(wᵢ, nstates(bp,i), μ[ein.|>idx], ψ[eout.|>idx], getT(bp), svd_trunc)
    for (j,e) = enumerate(eout)
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ - 1, nstates(bp, dst(e)), j)
        μj = orthogonalize_right!(mpem2(B); svd_trunc)
        sumlogzᵢ₂ⱼ += set_msg!(bp, μj, idx(e), damp, svd_trunc)
    end
    B = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bp.b[i] = B |> mpem2 |> marginalize
    logzᵢ += normalize!(bp.b[i])
    bp.f[i] = (dᵢ/2-1)*logzᵢ - (1/2)*sumlogzᵢ₂ⱼ
    nothing
end

# write message to destination after applying damping
function set_msg!(bp::MPBP{G,F,V,M2}, μj::M2, edge_id, damp, svd_trunc) where {G,F,V,M2}
    @assert 0 ≤ damp < 1
    μ_old = bp.μ[edge_id]
    logzᵢ₂ⱼ = normalize!(μj)
    if damp > 0 
        μj = _compose(x->x*damp/(1-damp), μj, μ_old)
        compress!(μj; svd_trunc)
        normalize!(μj)
    end
    bp.μ[edge_id] = μj
    logzᵢ₂ⱼ
end

# adds a further transition xᵢᵗ->xᵢᵗ⁺¹ with probability `p` and rescales all other
#  transitions by `1-p`. Does nothing for `p=0`
struct DampedFactor{T<:RecursiveBPFactor,F<:Real} <: RecursiveBPFactor
    w :: T      # factor
    p :: F      # probability of staying in previous state        
    function DampedFactor(w::T, p::F) where {T<:RecursiveBPFactor,F<:Real}
        @assert 0 ≤ p ≤ 1
        new{T,F}(w, p)
    end
end

nstates(::Type{<:DampedFactor{T}}, l::Integer) where {T} = nstates(T, l)

@forward DampedFactor.w prob_xy, prob_yy

function (wᵢ::DampedFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    return (1-wᵢ.p)*(wᵢ.w(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) + wᵢ.p*(xᵢᵗ⁺¹ == xᵢᵗ)  
end

function prob_y(wᵢ::DampedFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    return (1-wᵢ.p)*(prob_y(wᵢ.w, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)) + wᵢ.p*(xᵢᵗ⁺¹ == xᵢᵗ)
end