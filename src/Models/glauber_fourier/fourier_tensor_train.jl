# Fourier basis functions
u_n(n::Int, P::Float64) = x -> exp(im*2π/P*n*x)

function offset_fourier_freqs(tensors::Vector{Array{F,N}}, ax::Vector{Int64}) where {F<:Complex, N}
    A = map(tensors) do Aᵗ
        for i in ax
            K = (size(Aᵗ)[i]-1)/2
            isinteger(K) ? K=Int(K) : throw(ArgumentError("Wrong dimension for axis of coefficients, got K=$(K)"))
            oldaxes = axes(Aᵗ)
            newaxes = [oldaxes[begin:i-1]..., -K:K, oldaxes[i+1:end]...]
            Aᵗ = OffsetArray(Aᵗ, newaxes...)
        end
        return Aᵗ
    end
end

"""
    fourierTensorTrain{tensors}

A function for creating a Fourier tensor train (i.e. a Tensor Train that approximates a function of continuous inputs in the Fourier basis)
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
- `ax` expresses the fourier-transformed axis
"""
function fourier_tensor_train(tensors::Vector{<:AbstractArray{F,N}};
    z=Logarithmic(one(abs(tensors[1][1]))), ax::Vector{Int64}=[3]) where {F<:Complex, N}
    N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
    return TensorTrain{F,N}(offset_fourier_freqs(tensors, ax); z)
end


function flat_fourier_tt(::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where {T<:Complex}
    flat_tt(T, bondsizes, q...)
end
flat_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...) = flat_fourier_tt(ComplexF64, bondsizes, q...)
flat_fourier_tt(d::Integer, L::Integer, q...) = flat_fourier_tt([1; fill(d, L-1); 1], q...)

function rand_fourier_tt(::Type{Tp}, bondsizes::AbstractVector{<:Integer}, q...) where {Tp<:Complex}
    tensors = rand_tt(Tp, bondsizes, q...).tensors
    fourier_tensor_train(tensors)
end
rand_fourier_tt(bondsizes::AbstractVector{<:Integer}, q...) = rand_fourier_tt(ComplexF64, bondsizes, q...)
rand_fourier_tt(d::Integer, L::Integer, q...) = rand_fourier_tt([1; fill(d, L-1); 1], q...)


"""
    fourier_tensor_train_spin(A::TensorTrain{F,N}, K::Int, d::Int, P::Float64, σ::Float64) where {F,N}

Computes a Fourier Tensor Train starting from a Tensor Train A, in which each tensor has three axes. The third index ``x`` of each tensor (the physical one) is assumed to represent the values of a spin ``s``, with the convention ``x=1 ⟺ s=-1`` and ``x=2 ⟺ s=+1``.
For the purpose of going to a continuous domain, each matrix ``Aᵗ[:,:,x]`` is approximated as a linear combination of gaussians with variance ``σ²``, centered in ``+1`` and ``-1``: ``Aᵗ[m,n,1] g(-1,σ²) + Aᵗ[m,n,2] g(+1,σ²)``.
"""
function fourier_tensor_train_spin(A::TensorTrain{U,N}, K::Int, P::Real, σ::Real) where {U,N}
    N<3 && throw(ArgumentError("Tensors must have at least three axes"))
    any(!=(2), [size(Aᵗ)[3] for Aᵗ in A]) && throw(ArgumentError("Third axis of tensors for spins must have dimension 2"))

    k = OffsetVector([2π/P*α for α in -K:K], -K:K)
    expon = OffsetVector([exp(-k[α]^2 / 2 * σ^2)/P for α in -K:K], -K:K)
    cos_kn = [expon[α] * cos(k[α]) for α in -K:K]
    sin_kn = [expon[α] * sin(k[α]) for α in -K:K]
    
    F = map(eachindex(A)) do t
        Aᵗ = reshape(A[t], size(A[t])[1:3]..., prod(size(A[t])[4:end]))
        @tullio Fᵗ[m,n,α,x] := (Aᵗ[m,n,1,x]+Aᵗ[m,n,2,x]) * cos_kn[α] + im * (Aᵗ[m,n,1,x]-Aᵗ[m,n,2,x]) * sin_kn[α]
        return reshape(Fᵗ, size(A[t])[1], size(A[t])[2], 2K+1, size(A[t])[4:end]...)
    end

    FTT = fourier_tensor_train(F, z=A.z)
    normalize_eachmatrix!(FTT)
    return FTT
end

"""
    marginals_fourier(A::TensorTrain{F,N}, P::Float64) where {F<:Complex,N}

Takes as input a Fourier Tensor Train `A` (with a single physical index) and computes the marginal distributions over the physical indices, returning a vector of functions. Each function represents the marginal distribution over the corresponding physical index, expressed in the continuous domain using the Fourier basis with period `P`.
"""
function marginals_fourier(A::TensorTrain{F,N}, P::Float64) where {F<:Complex,N}
    K = lastindex(A[1],3)
    firstindex(A[1],3) == -K || throw(ArgumentError("The Fourier basis functions must be centered at 0"))
    N>4 && throw(ArgumentError("Tensor train must have at most two physical indices"))

    N==4 && (A = marginalize(A))
    pF = marginals(A)
    
    map(pF) do pFᵗ
        pFᵗ = OffsetArray(pFᵗ, -K:K)
        norm2 = sum(abs2,pFᵗ)/P
        pFᵗ ./= sqrt(norm2)
        x -> sum(pFᵗ[n]*u_n(n,P)(x) for n in -K:K) |> real
    end
end
function marginals_fourier(A::TensorTrain{F,N}) where {F,N}
    error("The period of the basis function must be specified")
end