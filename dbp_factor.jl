include("utils.jl")
include("glauber.jl")

abstract type dBP_Factor; end

const q_glauber = 2

struct GlauberFactor{T<:Real} <: dBP_Factor
    J :: Vector{T}
    h :: T
end

function GlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    GlauberFactor(J.*β, h*β)
end

function (fᵢ::GlauberFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.J)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.J))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.h)
    exp( -E ) / (2cosh(E))
end

# construct an array of GlauberFactors corresponding to gl
function glauber_factors(ising::Ising, T::Integer)
    map(1:nv(ising.g)) do i
        ei = outedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = GlauberFactor(J, h, ising.β)
        fill(wᵢᵗ, T)
    end
end



