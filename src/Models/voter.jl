struct VoterFactor{T<:Real}  <: BPFactor 
    J :: Vector{T}      
end

function (fᵢ::VoterFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.J)

    isempty(fᵢ.J) && return 0.5

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.J))
    z = sum(abs, fᵢ.J)
    p = 0.5 * (1 - potts2spin(xᵢᵗ⁺¹) * hⱼᵢ / z)
    return p
end