const RECOVERED = 3

struct SIRSFactor{T<:AbstractFloat} <: RecursiveBPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    σ :: T  # deimmunization rate
    α :: T  # auto-infection rate
    function SIRSFactor(λ::T, ρ::T, σ::T; α=zero(T))  where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        @assert 0 ≤ σ ≤ 1
        @assert 0 ≤ α ≤ 1
        new{T}(λ, ρ, σ, α)
    end
end

# the accumulated variable is still binary
nstates(::SIRSFactor, l::Integer) = l == 0 ? 1 : 2


function mpbp(sirs::SIRS{T,N,F}; kw...) where {T,N,F}
    g = IndexedBiDiGraph(sirs.g.A)
    w = sirs_factors(sirs)
    return mpbp(g, w, fill(3, nv(g)), T, ϕ=sirs.ϕ, ψ=sirs.ψ; kw...)
end

function prob_y(wᵢ::SIRSFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    @unpack λ, ρ, σ, α = wᵢ
    w = (yᵗ == SUSCEPTIBLE) * (1 - α)
    if xᵢᵗ⁺¹ == INFECTIOUS
        return (xᵢᵗ == INFECTIOUS) * (1 - ρ) + (xᵢᵗ == SUSCEPTIBLE) * (1 - w) 
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ == RECOVERED) * σ + (xᵢᵗ == SUSCEPTIBLE) * w
    else #if xᵢᵗ⁺¹ == RECOVERED
        return (xᵢᵗ == INFECTIOUS) * ρ  + (xᵢᵗ == RECOVERED) * (1 - σ)
    end
end

function prob_xy(wᵢ::SIRSFactor, yₖ, xₖ, xᵢ)
    @unpack λ = wᵢ
    (yₖ == INFECTIOUS)*λ*(xₖ==INFECTIOUS) + (yₖ == SUSCEPTIBLE)*(1-λ*(xₖ==INFECTIOUS))
end

prob_yy(wᵢ::SIRSFactor, y, y1, y2, xᵢ) = 1.0*((y == INFECTIOUS) == ((y1 == INFECTIOUS) || (y2 == INFECTIOUS)))