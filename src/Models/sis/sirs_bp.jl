const SUSCEPTIBLE = 1 
const INFECTED = 2
const RECOVERED = 3

struct SIRSFactor{T<:AbstractFloat} <: RecursiveBPFactor
    λ :: T  # infection rate
    ρ :: T  # recovery rate
    σ :: T  # deimmunization rate
    function SIRSFactor(λ::T, ρ::T, σ::T)  where {T<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        @assert 0 ≤ σ ≤ 1
        new{T}(λ, ρ, σ)
    end
end

# the accumulated variable is still binary
nstates(::Type{<:SIRSFactor}, l::Integer) = l == 0 ? 1 : 2


function mpbp(sirs::SIRS{T,N,F}; kw...) where {T,N,F}
    sirs_ = deepcopy(sirs)
    g = IndexedBiDiGraph(sirs_.g.A)
    w = sirs_factors(sirs_)
    return mpbp(g, w, fill(3, nv(g)), T, ϕ=sirs_.ϕ, ψ=sirs_.ψ; kw...)
end

function prob_y(wᵢ::SIRSFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    @unpack λ, ρ, σ = wᵢ
    w = (yᵗ == SUSCEPTIBLE)
    if xᵢᵗ⁺¹ == INFECTED
        return (xᵢᵗ == INFECTED) * (1 - ρ) + (xᵢᵗ == SUSCEPTIBLE) * (1 - w) 
    elseif xᵢᵗ⁺¹ == SUSCEPTIBLE
        return (xᵢᵗ == RECOVERED) * σ + (xᵢᵗ == SUSCEPTIBLE) * w
    else #if xᵢᵗ⁺¹ == RECOVERED
        return (xᵢᵗ == INFECTED) * ρ  + (xᵢᵗ == RECOVERED) * (1 - σ)
    end
end

function prob_xy(wᵢ::SIRSFactor, yₖ, xₖ, xᵢ)
    @unpack λ = wᵢ
    (yₖ == INFECTED)*λ*(xₖ==INFECTED) + (yₖ == SUSCEPTIBLE)*(1-λ*(xₖ==INFECTED))
end

prob_yy(wᵢ::SIRSFactor, y, y1, y2, xᵢ) = 1.0*((y == INFECTED) == ((y1 == INFECTED) || (y2 == INFECTED)))