using UnPack
include("utils.jl")

```
Factor for the factor graph of a model solvable with MPdBP.
Any `dBP_Factor` subtype must implement a functor that computes the Boltzmann
contribution to the joint probability
```
abstract type dBP_Factor; end

const q_glauber = 2

struct GlauberFactor{T<:Real} <: dBP_Factor
    βJ :: Vector{T}      
    βh :: T
end

function GlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    GlauberFactor(J.*β, h*β)
end

function (fᵢ::GlauberFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.βJ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.βh)
    exp( -E ) / (2cosh(E))
end

const S = 1
const I = 2
const q_sis = 2

struct SISFactor{T<:AbstractFloat} <: dBP_Factor
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

    ( xᵢᵗ == I && xᵢᵗ⁺¹ == S ) && return κ
    ( xᵢᵗ == I && xᵢᵗ⁺¹ == I ) && return 1 - κ 
    if xᵢᵗ == S
        p = (1-λ)^sum( xⱼᵗ == I for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == S
            return p
        elseif xᵢᵗ⁺¹ == I
            return 1 - p
        end
    end
    error("Shouldn't end up here")
    0.0
end