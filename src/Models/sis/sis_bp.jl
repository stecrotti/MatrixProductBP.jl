const SUSCEPTIBLE = 1
const INFECTED = 2
const q_sis = 2

struct SISFactor{T<:AbstractFloat} <: SimpleBPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    function SISFactor(λ::T, ρ::T) where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        new{T}(λ, ρ)
    end
end

getq(::Type{<:SISFactor}) = q_sis

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
        wᵢ::Vector{U}, l::Integer) where {q,T,F,U<:SISFactor}
    @assert q==q_sis
    AA = Vector{Array{F,4}}(undef, T+1)

    λ = wᵢ[1].λ     # can be improved

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
function f_bp_partial_ij(A::MPEM2{q,T,F}, pᵢ⁰, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer; prob = prob_ijy(U)) where {q,T,F,U<:SISFactor}

    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)

    for xᵢ⁰ in 1:q
        for xᵢ¹ in 1:q
            for y⁰ in 1:q
                for xⱼ⁰ in 1:q
                    p = prob(xᵢ¹, xᵢ⁰,xⱼ⁰, y⁰, wᵢ[begin].λ, wᵢ[begin].ρ)
                    B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,y⁰,xᵢ⁰]
                end
            end
        end
        B⁰[xᵢ⁰,:,:,:,:] .*= pᵢ⁰[xᵢ⁰]  * ϕᵢ[begin][xᵢ⁰] 
    end
    B[begin] = B⁰

    for t in 1:T-1
        Aᵗ = A[begin+t]
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xⱼᵗ in 1:q
                    for yᵗ in 1:q
                        p = prob(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, wᵢ[t+1].λ, wᵢ[t+1].ρ)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,yᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[xᵢᵗ,:,:,:,:] *= ϕᵢ[t+1][xᵢᵗ]
        end
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        B[begin+t] = Bᵗ
    end

    Aᵀ = A[end]
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)

    for xᵢᵀ in 1:q
        for xᵢᵀ⁺¹ in 1:q
            for xⱼᵀ in 1:q
                for yᵀ in 1:q
                    Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,yᵀ,xᵢᵀ]
                end
            end
        end
        Bᵀ[xᵢᵀ,:,:,:,:] *= ϕᵢ[end][xᵢᵀ]
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

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