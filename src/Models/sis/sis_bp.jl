const SUSCEPTIBLE = 1
const INFECTED = 2

struct SISFactor{T<:AbstractFloat} <: SimpleBPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    function SISFactor(λ::T, ρ::T) where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        new{T}(λ, ρ)
    end
end

getq(::Type{<:SISFactor}) = 2

function (fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

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
    return mpbp(g, w, T, p⁰=sis_.p⁰, ϕ=sis_.ϕ, ψ=sis_.ψ; kw...)
end

idx_to_value(x::Integer, ::Type{<:SISFactor}) = x - 1

function prob_partial_msg_sis(yₖ, yₖ₁, xₖ, λ)
    if yₖ == INFECTED
        return 1 - (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    elseif yₖ == SUSCEPTIBLE
        return (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    end
end


# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q,T,F}, 
        wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {q,T,F,U<:SISFactor}
    @assert q == 2
    AA = Vector{Array{F,4}}(undef, T+1)

    for t in eachindex(AA)
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        AAᵗ = zeros(nrows, ncols, q, q)
        if t ≤ T
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg_sis(yₗᵗ,yₗ₁ᵗ,xₗᵗ,wᵢ[$t].λ) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        else
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = 1/q * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        end
        AA[t] = AAᵗ
    end

    return MPEM2(AA)
end


# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij(A::MPEM2{q,T,F}, pᵢ⁰, wᵢ::Vector{U}, ϕᵢ, 
    d::Integer; prob = prob_ijy(U)) where {q,T,F,U<:SISFactor}
    B = [zeros(q, q, size(a,1), size(a,2), q) for a in A]
    A⁰, B⁰ = A[begin], B[begin]
    @tullio B⁰[xᵢ⁰,xⱼ⁰,1,n,xᵢ¹] = prob(xᵢ¹,xᵢ⁰,xⱼ⁰,y⁰,wᵢ[1].λ,wᵢ[1].ρ)*A⁰[1,n,y⁰,xᵢ⁰]*ϕᵢ[1][xᵢ⁰]*pᵢ⁰[xᵢ⁰]
    for t in 2:T
        Aᵗ,Bᵗ = A[t], B[t]
        @tullio Bᵗ[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹] = prob(xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,wᵢ[$t].λ,wᵢ[$t].ρ)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[xᵢᵀ,xⱼᵀ,m,n,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && println("NaN in tensor train")
    return MPEM3(B)
end


function prob_ijy(::Type{<:SISFactor})
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
end

function prob_ijy_dummy(U::Type{<:SISFactor})
    # neighbor j is susceptible -> does nothing
    function prob_ijy_dummy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, ρ)
        xⱼᵗ = SUSCEPTIBLE
        return prob_ijy(U)(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, ρ)
    end
end

function sis_infinite_graph(T::Integer, k::Integer, pᵢ⁰, λ::Real, ρ::Real;
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true)
    wᵢ = fill(SISFactor(λ, ρ), T)
    A, maxiter, Δs = iterate_bp_infinite_graph(T, k, pᵢ⁰, wᵢ; 
        svd_trunc, maxiter, tol, showprogress)
end
