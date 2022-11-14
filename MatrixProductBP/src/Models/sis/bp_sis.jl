const SUSCEPTIBLE = 1
const INFECTED = 2
const q_sis = 2

struct SISFactor{T<:AbstractFloat} <: BPFactor
    λ :: T  # infection rate
    κ :: T  # recovery rate
    function SISFactor(λ::T, κ::T) where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ κ ≤ 1
        new{T}(λ, κ)
    end
end

function (fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_sis
    @assert all(x ∈ 1:q_sis for x in xₙᵢᵗ)

    @unpack λ, κ = fᵢ

    ( xᵢᵗ == INFECTED && xᵢᵗ⁺¹ == SUSCEPTIBLE ) && return κ
    ( xᵢᵗ == INFECTED && xᵢᵗ⁺¹ == INFECTED ) && return 1 - κ 
    if xᵢᵗ == SUSCEPTIBLE
        p = (1-λ)^sum( xⱼᵗ == INFECTED for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTED
            return 1 - p
        end
    end
    error("Shouldn't end up here")
    0.0
end

function mpdbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    g = IndexedBiDiGraph(sis.g.A)
    w = sis_factors(sis)
    return mpdbp(g, w, q_sis, T, p⁰=sis.p⁰, ϕ=sis.ϕ, ψ=sis.ψ; kw...)
end

# compute outgoing message efficiently for any degree
# return a `MPMEM3` just like `f_bp`
function f_bp_sis(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F}
    
    λ = wᵢ[1].λ; @assert all(wᵢᵗ.λ == λ for wᵢᵗ in wᵢ)
    κ = wᵢ[1].κ; @assert all(wᵢᵗ.κ == κ for wᵢᵗ in wᵢ)

    # initialize recursion
    M = reshape([1.0 1; 0 0], (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    for k in eachindex(A)
        k == j && continue
        mᵢⱼₗ₁ = f_bp_partial_sis(A[k], mᵢⱼₗ₁, λ)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij_sis(mᵢⱼₗ₁, λ, κ, pᵢ⁰, ϕᵢ)
    return B
end

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial_sis(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q,T,F}, 
        λ::Real) where {q,T,F}
    @assert q==q_sis
    AA = Vector{Array{F,4}}(undef, T+1)

    for t in eachindex(AA)
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        AAᵗ = zeros(nrows, ncols, q, q)
        for yₗᵗ in 1:q
            for xᵢᵗ in 1:q
                for yₗ₁ᵗ in 1:q
                    for xₗᵗ in 1:q
                        p = prob_partial_msg_sis(yₗᵗ, yₗ₁ᵗ, xₗᵗ, λ)
                        AAᵗ[:,:,yₗᵗ,xᵢᵗ] .+= p * Aᵗ[:,:,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] 
                    end
                end
            end
        end
        AA[t] = AAᵗ
    end

    return MPEM2(AA)
end

# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij_sis(A::MPEM2{q,T,F}, λ::Real, κ::Real, 
        pᵢ⁰, ϕᵢ) where {q,T,F}

    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)

    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for y⁰ in 1:q
            for xⱼ⁰ in 1:q
                p = prob_ijy_sis(xᵢ¹, xᵢ⁰,xⱼ⁰, y⁰, λ, κ)
                B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,y⁰,xᵢ⁰]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
    end
    B[begin] = B⁰

    for t in 1:T-1
        Aᵗ = A[begin+t]
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ⁺¹ in 1:q
            for xᵢᵗ in 1:q
                for xⱼᵗ in 1:q
                    for yᵗ in 1:q
                        p = prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, κ)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,yᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+1][xᵢᵗ⁺¹]
        end
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        B[begin+t] = Bᵗ
    end

    Aᵀ = A[end]
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)

    for xᵢᵀ⁺¹ in 1:q
        for xᵢᵀ in 1:q
            for xⱼᵀ in 1:q
                for yᵀ in 1:q
                    Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,yᵀ,xᵢᵀ]
                end
            end
        end
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

    return MPEM3(B)
end

function prob_partial_msg_sis(yₖ, yₖ₁, xₖ, λ)
    if yₖ == INFECTED
        return 1 - (yₖ₁==S)*(1-λ*(xₖ==I))
    elseif yₖ == SUSCEPTIBLE
        return (yₖ₁==S)*(1-λ*(xₖ==I))
    end
end

function prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, κ)
    z = 1 - λ*(xⱼᵗ==I)
    w = (yᵗ==S)
    if xᵢᵗ⁺¹ == I
        return (xᵢᵗ==I) * (1 - κ) + (xᵢᵗ==S) * (1 - z * w) 
    elseif xᵢᵗ⁺¹ == S
        return (xᵢᵗ==I) * κ + (xᵢᵗ==S) * z * w
    end
    error("shouldn't be here")
    return -Inf
end

function onebpiter!(bp::MPBP{q,T,F,<:SISFactor}, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    zᵢ = 1.0
    for (j_ind, e_out) in enumerate(eout)
        B = f_bp_sis(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        zᵢ₂ⱼ = normalize!(μ[idx(e_out)])
        zᵢ *= zᵢ₂ⱼ
    end
    dᵢ = length(ein)
    return zᵢ ^ (1 / dᵢ)
end