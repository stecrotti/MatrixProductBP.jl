const SUSCEPTIBLE = 1
const INFECTED = 2
const q_sis = 2

struct SISFactor{T<:AbstractFloat} <: BPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    function SISFactor(λ::T, ρ::T) where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        new{T}(λ, ρ)
    end
end

function (fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_sis
    @assert all(x ∈ 1:q_sis for x in xₙᵢᵗ)

    @unpack λ, ρ = fᵢ

    ( xᵢᵗ == INFECTED && xᵢᵗ⁺¹ == SUSCEPTIBLE ) && return ρ
    ( xᵢᵗ == INFECTED && xᵢᵗ⁺¹ == INFECTED ) && return 1 - ρ 
    if xᵢᵗ == SUSCEPTIBLE
        p = (1-λ)^sum( xⱼᵗ == INFECTED for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTED
            return 1 - p
        end
    end
    error("Shouldn't end up here")
    -Inf
end

function mpbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    sis_ = deepcopy(sis)
    g = IndexedBiDiGraph(sis_.g.A)
    w = sis_factors(sis_)
    return mpbp(g, w, q_sis, T, p⁰=sis_.p⁰, ϕ=sis_.ϕ, ψ=sis_.ψ; kw...)
end

idx_to_value(x::Integer, ::Type{<:SISFactor}) = x - 1

# compute outgoing message efficiently for any degree
# return a `MPMEM3` just like `f_bp`
function f_bp_sis(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F}
    
    λ = wᵢ[1].λ; @assert all(wᵢᵗ.λ == λ for wᵢᵗ in wᵢ)
    ρ = wᵢ[1].ρ; @assert all(wᵢᵗ.ρ == ρ for wᵢᵗ in wᵢ)

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
    B = f_bp_partial_ij_sis(mᵢⱼₗ₁, λ, ρ, pᵢ⁰, ϕᵢ)
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
function f_bp_partial_ij_sis(A::MPEM2{q,T,F}, λ::Real, ρ::Real, 
        pᵢ⁰, ϕᵢ) where {q,T,F}

    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)

    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for y⁰ in 1:q
            for xⱼ⁰ in 1:q
                p = prob_ijy_sis(xᵢ¹, xᵢ⁰,xⱼ⁰, y⁰, λ, ρ)
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
                        p = prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, ρ)
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
        return 1 - (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    elseif yₖ == SUSCEPTIBLE
        return (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    end
end

function prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, ρ)
    z = 1 - λ*(xⱼᵗ==INFECTED)
    w = (yᵗ==SUSCEPTIBLE)
    if xᵢᵗ⁺¹ == INFECTED
        return (xᵢᵗ==INFECTED) * (1 - ρ) + (xᵢᵗ==SUSCEPTIBLE) * (1 - z * w) 
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ==INFECTED) * ρ + (xᵢᵗ==SUSCEPTIBLE) * z * w
    end
    error("shouldn't be here")
    return -Inf
end

function onebpiter!(bp::MPBP{q,T,F,<:SISFactor}, i::Integer; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}

    _onebpiter!(bp, i, f_bp_sis; svd_trunc)
end