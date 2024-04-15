const SUSCEPTIBLE = 1 
const INFECTIOUS = 2

struct SIS_heterogeneousFactor{T<:AbstractFloat} <: RecursiveBPFactor
    λ :: Vector{T}  # infection rates
    ρ :: T          # recovery rate
    function SIS_heterogeneousFactor(λ::Vector{T}, ρ::T) where {T<:AbstractFloat}
        @assert all(0 ≤ λⱼᵢ ≤ 1 for λⱼᵢ in λ)
        @assert 0 ≤ ρ ≤ 1
        new{T}(λ, ρ)
    end
end

# the accumulated variable is still binary
nstates(::SIS_heterogeneousFactor, l::Int64) = (l == 0 ? 1 : 2)

function (fᵢ::SIS_heterogeneousFactor)(xᵢᵗ⁺¹::Integer, xᵗ::AbstractVector{<:Integer}, g::IndexedGraph, i::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xᵗ)

    @unpack λ, ρ = fᵢ

    if xᵗ[i] == INFECTIOUS
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return ρ
        else
            return 1 - ρ
        end
    else
        p = 1.0
        for j in neighbors(g,i)
            p *= (1-λ[j])*(xᵗ[j] == INFECTIOUS)
        end
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTIOUS
            return 1 - p
        end
    end
end

function mpbp(sis::SIS_heterogeneous{T,N,F}; kw...) where {T,N,F}
    sis_ = deepcopy(sis)
    g = IndexedBiDiGraph(sis_.g.A)
    w = sis_heterogeneous_factors(sis_)
    return mpbp(g, w, fill(2, nv(g)), T, ϕ=sis_.ϕ, ψ=sis_.ψ; kw...)
end

function periodic_mpbp(sis::SIS_heterogeneous{T,N,F}; kw...) where {T,N,F}
    sis_ = deepcopy(sis)
    g = IndexedBiDiGraph(sis_.g.A)
    w = sis_heterogeneous_factors(sis_)
    return periodic_mpbp(g, w, fill(2, nv(g)), T, ϕ=sis_.ϕ, ψ=sis_.ψ; kw...)
end

# neighbor j is susceptible -> does nothing
function prob_y(wᵢ::SIS_heterogeneousFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    @unpack λ, ρ = wᵢ
    if xᵢᵗ⁺¹ == INFECTIOUS
        return (xᵢᵗ==INFECTIOUS) * (1 - ρ) + (xᵢᵗ==SUSCEPTIBLE) * (yᵗ == INFECTIOUS)
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ==INFECTIOUS) * ρ + (xᵢᵗ==SUSCEPTIBLE) * (yᵗ == SUSCEPTIBLE)
    end
end

function prob_xy(wᵢ::SIS_heterogeneousFactor, yₖ, xₖ, xᵢ,k)
    @unpack λ = wᵢ
    (yₖ == INFECTIOUS)*λ[k]*(xₖ==INFECTIOUS) + (yₖ == SUSCEPTIBLE)*(1-λ[k]*(xₖ==INFECTIOUS))
end

prob_yy(wᵢ::SIS_heterogeneousFactor, y, y1, y2, xᵢ) = 1.0*((y == INFECTIOUS) == ((y1 == INFECTIOUS) || (y2 == INFECTIOUS)))