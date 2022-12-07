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

    if xᵢᵗ == INFECTED
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return ρ
        else
            return 1 - ρ 
        end
    else
        p = (1-λ)^sum(xⱼᵗ == INFECTED for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTED
            return 1 - p
        end
    end
end

function mpbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    sis_ = deepcopy(sis)
    g = IndexedBiDiGraph(sis_.g.A)
    w = sis_factors(sis_)
    return mpbp(g, w, T, ϕ=sis_.ϕ, ψ=sis_.ψ; kw...)
end

idx_to_value(x::Integer, ::Type{<:SISFactor}) = x - 1

function prob_partial_msg(wᵢ::SISFactor, yₖ, yₖ₁, xₖ, l)
    @unpack λ = wᵢ
    if yₖ == INFECTED
        return 1 - (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    elseif yₖ == SUSCEPTIBLE
        return (yₖ₁==SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
    end
end


# # compute message m(i→j, l) from m(i→j, l-1) 
# # returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
# function f_bp_partial(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q,T,F}, 
#         wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {q,T,F,U<:SISFactor}
#     @assert q == 2
#     map(1:T+1) do t
#         Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
#         AAᵗ = zeros(size(Aᵗ, 1), size(Aᵗ, 2), q, q)
#         if t ≤ T
#             @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg_sis(yₗᵗ,yₗ₁ᵗ,xₗᵗ,wᵢ[$t].λ) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
#         else
#             @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = 1/q * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
#         end
#     end |> MPEM2
# end


# # compute m(i→j) from m(i→j,d)
# function f_bp_partial_ij(A::MPEM2{q,T,F}, wᵢ::Vector{U}, ϕᵢ, 
#     d::Integer; prob = prob_ijy) where {q,T,F,U<:SISFactor}
#     B = [zeros(q, q, size(a,1), size(a,2), q) for a in A]
#     for t in 1:T
#         Aᵗ,Bᵗ = A[t], B[t]
#         @tullio Bᵗ[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
#     end
#     Aᵀ,Bᵀ = A[end], B[end]
#     @tullio Bᵀ[xᵢᵀ,xⱼᵀ,m,n,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
#     any(any(isnan, b) for b in B) && println("NaN in tensor train")
#     return MPEM3(B)
# end


function prob_ijy(wᵢ::SISFactor, xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, d)
    @unpack λ, ρ = wᵢ
    z = 1 - λ*(xⱼᵗ==INFECTED)
    w = (yᵗ==SUSCEPTIBLE)
    if xᵢᵗ⁺¹ == INFECTED
        return (xᵢᵗ==INFECTED) * (1 - ρ) + (xᵢᵗ==SUSCEPTIBLE) * (1 - z * w) 
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ==INFECTED) * ρ + (xᵢᵗ==SUSCEPTIBLE) * z * w
    end
end

# neighbor j is susceptible -> does nothing
function prob_ijy_dummy(wᵢ::SISFactor, xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, d)
    xⱼᵗ = SUSCEPTIBLE
    return prob_ijy(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, d)
end

function sis_infinite_graph(T::Integer, k::Integer, ϕᵢ, λ::Real, ρ::Real;
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true)
    wᵢ = fill(SISFactor(λ, ρ), T)
    A, iters, Δs = iterate_bp_infinite_graph(T, k, wᵢ, ϕᵢ;
        svd_trunc, maxiter, tol, showprogress)
end
