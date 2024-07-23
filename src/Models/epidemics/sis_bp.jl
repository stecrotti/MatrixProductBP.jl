const SUSCEPTIBLE = 1 
const INFECTIOUS = 2

struct SISFactor{T<:AbstractFloat} <: RecursiveBPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    α :: T  # auto-infection rate

    function SISFactor(λ::T, ρ::T; α=zero(T)) where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        @assert 0 ≤ α ≤ 1
        new{T}(λ, ρ, α)
    end
end

# the accumulated variable is still binary
nstates(::SISFactor, l::Integer) = l == 0 ? 1 : 2

function (fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    @unpack λ, ρ, α = fᵢ

    if xᵢᵗ == INFECTIOUS
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return ρ
        else
            return 1 - ρ 
        end
    else
        p = (1-α)*(1-λ)^sum(xⱼᵗ == INFECTIOUS for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTIOUS
            return 1 - p
        end
    end
end

function mpbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    g = IndexedBiDiGraph(sis.g.A)
    w = sis_factors(sis)
    return mpbp(g, w, fill(2, nv(g)), T, ϕ=sis.ϕ, ψ=sis.ψ; kw...)
end

function periodic_mpbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    g = IndexedBiDiGraph(sis.g.A)
    w = sis_factors(sis)
    return periodic_mpbp(g, w, fill(2, nv(g)), T, ϕ=sis.ϕ, ψ=sis.ψ; kw...)
end

function mpbp_stationary(sis::SIS{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(sis.g.A)
    w = sis_factors(sis)
    return mpbp_stationary(g, w, fill(2, nv(g)); ϕ=sis.ϕ, ψ=sis.ψ, kw...)
end

# neighbor j is susceptible -> does nothing
function prob_y(wᵢ::SISFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    @unpack λ, ρ, α = wᵢ
    xⱼᵗ = SUSCEPTIBLE
    z = 1 - λ*(xⱼᵗ == INFECTIOUS)
    w = (yᵗ == SUSCEPTIBLE) * (1 - α)
    if xᵢᵗ⁺¹ == INFECTIOUS
        return (xᵢᵗ==INFECTIOUS) * (1 - ρ) + (xᵢᵗ==SUSCEPTIBLE) * (1 - z * w) 
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ==INFECTIOUS) * ρ + (xᵢᵗ==SUSCEPTIBLE) * z * w
    end
end

function prob_xy(wᵢ::SISFactor, yₖ, xₖ, xᵢ)
    @unpack λ = wᵢ
    (yₖ == INFECTIOUS)*λ*(xₖ==INFECTIOUS) + (yₖ == SUSCEPTIBLE)*(1-λ*(xₖ==INFECTIOUS))
end

prob_yy(wᵢ::SISFactor, y, y1, y2, xᵢ) = 1.0*((y == INFECTIOUS) == ((y1 == INFECTIOUS) || (y2 == INFECTIOUS)))