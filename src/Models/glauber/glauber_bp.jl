struct GenericGlauberFactor{T<:Real}  <: BPFactor 
    βJ :: Vector{T}      
    βh :: T
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
    return 1 / (1 + exp(2E))
end

struct HomogeneousGlauberFactor{T<:Real} <: RecursiveBPFactor 
    βJ :: T     
    βh :: T
end

function HomogeneousGlauberFactor(J::T, h::T, β::T) where {T<:Real}
    HomogeneousGlauberFactor(J*β, h*β)
end

# the sum of `l` spins can assume `l+1` values
nstates(::HomogeneousGlauberFactor, l::Integer) = l + 1

# ignore neighbor because it doesn't exist
function prob_y(wᵢ::HomogeneousGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, zᵗ, d)
    @unpack βJ, βh = wᵢ
    yᵗ = 2 * zᵗ - 2 - d
    hⱼᵢ = βJ * yᵗ + βh
    E = - potts2spin(xᵢᵗ⁺¹) * hⱼᵢ
    return 1 / (1 + exp(2E))
end

prob_xy(wᵢ::HomogeneousGlauberFactor, yₖ, xₖ, xᵢ) = (yₖ != xₖ)
prob_yy(wᵢ::HomogeneousGlauberFactor, y, y1, y2, xᵢ) = (y == y1 + y2 - 1)

function (wᵢ::HomogeneousGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    hⱼᵢ = wᵢ.βJ * sum(potts2spin, xₙᵢᵗ; init=0.0)
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + wᵢ.βh)
    return 1 / (1 + exp(2E))
end

# Ising model with ±J interactions
struct PMJGlauberFactor{T<:Real} <: RecursiveBPFactor
    signs :: Vector{Int}
    βJ    :: T     
    βh    :: T 
end

function PMJGlauberFactor(signs::Vector{Int}, J::T, h::T, β::T) where {T<:Real}
    PMJGlauberFactor(signs, J*β, h*β)
end

nstates(::PMJGlauberFactor, d::Integer) = 2d + 1

function prob_y(wᵢ::PMJGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    @unpack βJ, βh = wᵢ
    hᵗ = yᵗ - d - 1
    βhⱼᵢ = βJ * hᵗ + βh
    E = - potts2spin(xᵢᵗ⁺¹) * βhⱼᵢ
    return 1 / (1 + exp(2E))
end

# yₖ = σₖ*sign(Jᵢₖ), but with xₖ ∈ {1,2}, yₖ ∈ {1,2,3}
prob_xy(wᵢ::PMJGlauberFactor, yₖ, xₖ, xᵢ, k) = (yₖ == potts2spin(xₖ)*wᵢ.signs[k] + 2)
prob_yy(wᵢ::PMJGlauberFactor, y, y1, y2, xᵢ) = (y == y1 + y2 - 1)

function (wᵢ::PMJGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    hⱼᵢ = wᵢ.βJ * sum( s * potts2spin(xⱼᵗ) for (xⱼᵗ,s) in zip(xₙᵢᵗ, wᵢ.signs); init=0.0)
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + wᵢ.βh)
    return 1 / (1 + exp(2E))
end


function mpbp(gl::Glauber{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors(gl.ising, T)
    ϕ = gl.ϕ
    ψ = pair_obs_undirected_to_directed(gl.ψ, gl.ising.g)
    return mpbp(g, w, fill(2, nv(g)), T; ϕ, ψ, kw...)
end

function periodic_mpbp(gl::Glauber{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors(gl.ising, T)
    ϕ = gl.ϕ
    ψ = pair_obs_undirected_to_directed(gl.ψ, gl.ising.g)
    return periodic_mpbp(g, w, fill(2, nv(g)), T; ϕ, ψ, kw...)
end


# construct an array of GlauberFactors corresponding to gl
# seems to be type stable
function glauber_factors(ising::Ising, T::Integer)
    β = ising.β
    map(1:nv(ising.g)) do i
        ei = inedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = if is_absJ_const(ising)
            Jᵢ = length(∂i) == 0 ? 0.0 : J[1] 
            if is_homogeneous(ising)
                HomogeneousGlauberFactor(Jᵢ, h, β)
            else
                PMJGlauberFactor(Int.(sign.(J)), β*abs(Jᵢ), β*h)
            end
        elseif all(isinteger, J)
            IntegerGlauberFactor(Int.(J), h, β)
        else
            GenericGlauberFactor(J, h, β)
        end
        fill(wᵢᵗ, T + 1)
    end
end

struct IntegerGlauberFactor{T<:Real}  <: RecursiveBPFactor 
    J :: Vector{Int}      
    h :: T
    β :: T
    K :: Int
end

IntegerGlauberFactor(J,h,β) = IntegerGlauberFactor(J, h, β, sum(abs,J) + 1)


nstates(w::IntegerGlauberFactor, l::Integer) = 2w.K-1

"P(xᵢᵗ⁺¹|xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ)
Might depend on the degree `dᵢ` because of a change of variable from 
    y ∈ {1,2,...} to its physical value, e.g. {-dᵢ,...,dᵢ} for Ising"
function prob_y(wᵢ::IntegerGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) 
    @unpack J, h, β, K = wᵢ
    hᵗ = yₙᵢᵗ - K
    βhⱼᵢ = β*(hᵗ + h)
    E = - potts2spin(xᵢᵗ⁺¹) * βhⱼᵢ
    return 1 / (1 + exp(2E))
end

"P(yₖᵗ| xₖᵗ, xᵢᵗ)"
prob_xy(wᵢ::IntegerGlauberFactor, yₖ, xₖ, xᵢ, k) = (yₖ == potts2spin(xₖ)*wᵢ.J[k] + wᵢ.K)
prob_yy(wᵢ::IntegerGlauberFactor, y, y1, y2, xᵢ) = (y + wᵢ.K == y1 + y2)
prob_y0(wᵢ::IntegerGlauberFactor, y, xᵢ) = y == wᵢ.K

function (wᵢ::IntegerGlauberFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
    xᵢᵗ::Integer)
    @unpack J, h, β, K = wᵢ
    hᵗ = sum(Jk*potts2spin(xk) for (Jk,xk) in zip(J, xₙᵢᵗ); init=0.0)
    βhⱼᵢ = β*(hᵗ + h)
    E = - potts2spin(xᵢᵗ⁺¹) * βhⱼᵢ
    return 1 / (1 + exp(2E))    
end
