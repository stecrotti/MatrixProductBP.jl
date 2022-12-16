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

nstates(::Type{<:SISFactor}) = 2

# the accumulated variable is still binary
nstates(::Type{<:SISFactor}, l::Integer) = 2

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