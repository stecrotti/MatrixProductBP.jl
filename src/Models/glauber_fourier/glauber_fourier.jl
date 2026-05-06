""""
For a `w::U` where `U<:RecursiveBPFactor`, outgoing messages are computed recursively
with Fourier representation of messages
"""
abstract type FourierBPFactor <: RecursiveBPFactor; end

struct FourierGlauberFactor{T<:Real} <: FourierBPFactor 
    J :: Vector{T}      # vector of couplings to (in) neighbours
    h :: T              # external field
    β :: T              # inverse temperature
    K::Integer          # number of Fourier modes
    σ::Float64          # width of Gaussian to approximate spins
    P::Float64          # Period
    scale::Float64      # scaling factor
    p :: Float64        # probability of staying in previous state
end
function FourierGlauberFactor(J::Vector{T}, h::T, β::T; K::Integer=100, σ::Float64=1/100, P::Float64=2.0, p::Float64=0.0) where {T<:Real}
    @assert 0.0 ≤ p ≤ 1.0
    d = length(J)
    scale = 1 + ceil(d/4) / (d+1)   # = (d + 1 + ceil(d/4)) / (d+1)
    return FourierGlauberFactor{T}(J, h, β, K, σ, P, scale, p)
end

function (fᵢ::FourierGlauberFactor)(xᵢᵗ⁺¹::Integer, 
    xₙᵢᵗ::AbstractVector{<:Integer}, 
    xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.J)

    hⱼᵢ = sum((Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.J)); init=0.0)
    E = - fᵢ.β * (potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.h))
    return fᵢ.p * (xᵢᵗ⁺¹==xᵢᵗ) + (1-fᵢ.p) / (1 + exp(2E))
end

function DampedFactor(w::FourierBPFactor, p::Float64)
    @assert 0.0 ≤ p ≤ 1.0
    return FourierGlauberFactor(w.J, w.h, w.β, w.K, w.σ, w.P, w.scale, p)
end

Base.convert(::Type{<:AbstractTensorTrain{F1,N}}, A::TT) where {F1<:Number,N,TT<:AbstractTensorTrain} = 
    TensorTrain([F1.(a) for a in A]; z = A.z)

function mpbp_fourier(bp::MPBP; K=100, σ=1/100, P=2.0, kw...)   # does not handle DampedFactor yet
    @unpack g, w, ϕ, ψ, μ = bp
    T = getT(bp)

    wnew = map(w) do wᵢ
        map(wᵢ) do wᵢᵗ
            if wᵢᵗ isa GenericGlauberFactor
                β = maximum(abs, wᵢᵗ.βJ)
                FourierGlauberFactor(wᵢᵗ.βJ ./ β, wᵢᵗ.βh / β, β; K, σ, P)
            else 
                wᵢᵗ
            end
        end
    end

    mpbp(ComplexF64, g, convert(Vector{Vector{mapreduce(eltype, typejoin, wnew)}}, wnew), fill(2, nv(g)), T; ϕ, ψ, kw...)
end