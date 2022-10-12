include("utils.jl")
include("exact/exact_glauber.jl")

abstract type dBP_Factor; end

const q_glauber = 2

struct GlauberFactor{T<:Real} <: dBP_Factor
    J :: Vector{T}
    h :: T
end

function (fᵢ::GlauberFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.J))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.h)
    exp( -E )
end

# construct an array of GlauberFactors corresponding to gl
function glauber_factors(gl::ExactGlauber{T,N,F}) where {T,N,F}
    map(1:N) do i
        ei = outedges(gl.ising.g, i)
        ∂i = idx.(ei)
        J = gl.ising.J[∂i]
        h = gl.ising.h[i]
        wᵢᵗ = GlauberFactor(J, h)
        fill(wᵢᵗ, T)
    end
end



