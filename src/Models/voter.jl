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
    p = 0.5 * (1 + potts2spin(xᵢᵗ⁺¹) * hⱼᵢ / z)
    return p
end

struct HomogeneousVoterFactor{T<:Real}  <: RecursiveBPFactor 
    J :: T      
end

nstates(::HomogeneousVoterFactor, l::Integer) = l + 1

function prob_y(wᵢ::HomogeneousVoterFactor, xᵢᵗ⁺¹, xᵢᵗ, zᵗ, d)
    d == 0 && return 0.5
    (; J) = wᵢ
    yᵗ = 2 * zᵗ - 2 - d
    return 0.5 * (1 + potts2spin(xᵢᵗ⁺¹) * J * yᵗ / d)
end

prob_xy(::HomogeneousVoterFactor, yₖ, xₖ, xᵢ) = (yₖ != xₖ)
prob_yy(::HomogeneousVoterFactor, y, y1, y2, xᵢ) = (y == y1 + y2 - 1)

function (fᵢ::HomogeneousVoterFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    isempty(xₙᵢᵗ) && return 0.5

    hⱼᵢ = fᵢ.J * mean(potts2spin, xₙᵢᵗ)
    p = 0.5 * (1 + potts2spin(xᵢᵗ⁺¹) * hⱼᵢ)
    return p
end
