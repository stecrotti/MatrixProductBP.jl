struct GenericGlauberFactor{T<:Real}  <: BPFactor 
    βJ :: Vector{T}      
    βh :: T
end

struct HomogeneousGlauberFactor{T<:Real} <: RecursiveBPFactor 
    βJ :: T     
    βh :: T
end

# the sum of `l` spins can assume `l+1` values
nstates(::Type{<:HomogeneousGlauberFactor}, l::Integer) = l + 1

function HomogeneousGlauberFactor(J::T, h::T, β::T) where {T<:Real}
    HomogeneousGlauberFactor(J*β, h*β)
end

function GenericGlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    GenericGlauberFactor(J.*β, h*β)
end

function (fᵢ::GenericGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.βJ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.βh)
    exp( -E ) / (2cosh(E))
end

function (fᵢ::HomogeneousGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    hⱼᵢ = fᵢ.βJ * sum(potts2spin, xₙᵢᵗ)
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.βh)
    exp( -E ) / (2cosh(E))
end

function mpbp(gl::Glauber{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors(gl.ising, T)
    ϕ = gl.ϕ
    ψ = pair_obs_undirected_to_directed(gl.ψ, gl.ising.g)
    return mpbp(g, w, fill(2, nv(g)), T; ϕ, ψ, kw...)
end


# construct an array of GlauberFactors corresponding to gl
# seems to be type stable
function glauber_factors(ising::Ising, T::Integer)
    map(1:nv(ising.g)) do i
        ei = outedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = if is_homogeneous(ising)
            HomogeneousGlauberFactor(J[1], h, ising.β)
        else
            GenericGlauberFactor(J, h, ising.β)
        end
        fill(wᵢᵗ, T + 1)
    end
end

# ignore neighbor because it doesn't exist
function prob_y(wᵢ::HomogeneousGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, zᵗ, d)
    @unpack βJ, βh = wᵢ
    yᵗ = 2 * zᵗ - 2 - d
    h = βJ * yᵗ + βh
    p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
    @assert 0 ≤ p ≤ 1
    p
end

prob_xy(wᵢ::HomogeneousGlauberFactor, yₖ, xₖ, xᵢ) = (yₖ != xₖ)
prob_yy(wᵢ::HomogeneousGlauberFactor, y, y1, y2, xᵢ) = (y == y1 + y2 - 1)