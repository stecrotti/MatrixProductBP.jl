struct GenericGlauberFactor{T<:Real}  <: BPFactor 
    βJ :: Vector{T}      
    βh :: T
end

struct HomogeneousGlauberFactor{T<:Real} <: SimpleBPFactor 
    βJ :: T     
    βh :: T
end

nstates(::Type{<:GenericGlauberFactor}) = 2
nstates(::Type{<:HomogeneousGlauberFactor}) = 2

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
    return mpbp(g, w, T; ϕ, ψ, kw...)
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
        fill(wᵢᵗ, T)
    end
end

idx_to_value(x::Integer, ::Type{<:GenericGlauberFactor}) = potts2spin(x)
idx_to_value(x::Integer, ::Type{<:HomogeneousGlauberFactor}) = potts2spin(x)

prob_partial_msg(wᵢ::HomogeneousGlauberFactor, zₗᵗ, zₗ₁ᵗ, xₗᵗ, l) = ( zₗᵗ == ( zₗ₁ᵗ + 2 - xₗᵗ ) )

# # the sum of l spins can be one of (l+1) values. We sort them increasingly and
# #  index them by z
# function _idx_map(l::Integer, z::Integer) 
#     @assert l ≥ 0
#     @assert z ∈ 1:(l+1)
#     return - l + 2*(z-1)
# end

# # compute message m(i→j, l) from m(i→j, l-1) 
# # returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
# function f_bp_partial(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q1,T,F}, 
#         wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {q,q1,T,F,U<:HomogeneousGlauberFactor}
#     @assert q == 2
#     AA = Vector{Array{F,4}}(undef, T+1)

#     for t in eachindex(AA)
#         Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
#         nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
#         AAᵗ = zeros(nrows, ncols, l+1, l+1)
#         for zₗᵗ in 1:(l+1)
#             for xᵢᵗ in 1:q
#                 for zₗ₁ᵗ in 1:l
#                     for xₗᵗ in 1:q
#                         p = prob_partial_msg(U, zₗᵗ, zₗ₁ᵗ, xₗᵗ)
#                         AAᵗ[:,:,zₗᵗ,xᵢᵗ] .+= p * Aᵗ[:,:,xᵢᵗ,xₗᵗ,zₗ₁ᵗ] * ψᵢₗ[t][xᵢᵗ,xₗᵗ]
#                     end
#                 end
#             end
#         end
#         AA[t] = AAᵗ
#     end

#     return MPEM2(AA)
# end

# # compute m(i→j) from m(i→j,d)
# function f_bp_partial_ij(A::MPEM2{Q,T,F}, wᵢ::Vector{U}, ϕᵢ, 
#         d::Integer; prob = prob_ijy) where {Q,T,F,U<:HomogeneousGlauberFactor}
#     q = getq(U)
#     B = Vector{Array{F,5}}(undef, T+1)

#     A⁰ = A[begin]
#     nrows = size(A⁰, 1); ncols = size(A⁰, 2)
#     B⁰ = zeros(q, q, nrows, ncols, q)

#     for xᵢ⁰ in 1:q
#         for xᵢ¹ in 1:q
#             for z⁰ in 1:(d+1)
#                 y⁰ = _idx_map(d, z⁰)
#                 for xⱼ⁰ in 1:q
#                     p = prob(wᵢ[begin], xᵢ¹, xⱼ⁰, z⁰, d)
#                     B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,z⁰,xᵢ⁰]
#                 end
#             end
#         end
#         B⁰[xᵢ⁰,:,:,:,:] .*= ϕᵢ[begin][xᵢ⁰]
#     end
#     B[begin] = B⁰

#     for t in 1:T-1
#         Aᵗ = A[begin+t]
#         nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
#         Bᵗ = zeros(q, q, nrows, ncols, q)

#         for xᵢᵗ in 1:q
#             for xᵢᵗ⁺¹ in 1:q
#                 for xⱼᵗ in 1:q
#                     for zᵗ in 1:(d+1)
#                         yᵗ = _idx_map(d, zᵗ)
#                         p = prob(wᵢ[t+1], xᵢᵗ⁺¹, xⱼᵗ, zᵗ, d)
#                         Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,zᵗ,xᵢᵗ]
#                     end
#                 end
#             end
#             Bᵗ[xᵢᵗ,:,:,:,:] *= ϕᵢ[t+1][xᵢᵗ]
#         end
#         any(isnan, Bᵗ) && println("NaN in tensor at time $t")
#         B[begin+t] = Bᵗ
#     end

#     Aᵀ = A[end]
#     nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
#     Bᵀ = zeros(q, q, nrows, ncols, q)

#     for xᵢᵀ in 1:q
#         for xᵢᵀ⁺¹ in 1:q
#             for xⱼᵀ in 1:q
#                 for zᵀ in 1:(d+1)
#                     Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,zᵀ,xᵢᵀ]
#                 end
#             end
#         end
#         Bᵀ[xᵢᵀ,:,:,:,:] *= ϕᵢ[end][xᵢᵀ]
#     end
#     B[end] = Bᵀ
#     any(isnan, Bᵀ) && println("NaN in tensor at time $T")

#     return MPEM3(B)
# end


function prob_ijy(wᵢ::HomogeneousGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, zᵗ, d)
    @unpack βJ, βh = wᵢ
    yᵗ = 2 * zᵗ - 2 - d
    h = βJ * (potts2spin(xⱼᵗ) + yᵗ) + βh
    p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
    @assert 0 ≤ p ≤ 1
    p
end


# ignore neighbor because it doesn't exist
function prob_ijy_dummy(wᵢ::HomogeneousGlauberFactor, xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, zᵗ, d)
    @unpack βJ, βh = wᵢ
    yᵗ = 2 * zᵗ - 2 - d
    h = βJ * yᵗ + βh
    p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
    @assert 0 ≤ p ≤ 1
    p
end