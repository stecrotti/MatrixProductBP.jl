const AbstractMPEM1{F} = AbstractTensorTrain{F, 3}
const MPEM1{F} = TensorTrain{F, 3}
const PeriodicMPEM1{F} = PeriodicTensorTrain{F, 3}

# construct a flat mpem with given bond dimensions
flat_mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = flat_tt(bondsizes, q)
flat_periodic_mpem1(q::Int, T::Int; d::Int=2, bondsizes=fill(d, T+1)) = flat_periodic_tt(bondsizes, q)

# construct a random mpem with given bond dimensions
rand_mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_tt(bondsizes, q)
rand_periodic_mpem1(q::Int, T::Int; d::Int=2, bondsizes=fill(d, T+1)) = rand_periodic_tt(bondsizes, q)

nstates(A::AbstractMPEM1) = size(A[1], 3)

const AbstractMPEM2{F} = AbstractTensorTrain{F, 4}
const MPEM2{F} = TensorTrain{F, 4}
const PeriodicMPEM2{F} = PeriodicTensorTrain{F, 4}

# construct a flat mpem with given bond dimensions
flat_mpem2(q1::Int, q2::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = flat_tt(bondsizes, q1, q2)
flat_periodic_mpem2(q1::Int, q2::Int, T::Int; d::Int=2, bondsizes=fill(d, T+1)) = flat_periodic_tt(bondsizes, q1, q2)

# construct a flat mpem with given bond dimensions
rand_mpem2(q1::Int, q2::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_tt(bondsizes, q1, q2)
rand_periodic_mpem2(q1::Int, q2::Int, T::Int; d::Int=2, bondsizes=fill(d, T+1)) = rand_periodic_tt(bondsizes, q1, q2)

function marginalize(A::MPEM2{F}) where F
    MPEM1{F}([@tullio b[m, n, xi] := a[m, n, xi, xj] for a in A]; z = A.z)
end
function marginalize(A::PeriodicMPEM2{F}) where F
    PeriodicMPEM1{F}([@tullio b[m, n, xi] := a[m, n, xi, xj] for a in A]; z = A.z)
end

# Matrix [Bᵗᵢⱼ(xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 5-array B[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
# The last matrix should have the same values no matter what xᵢᵀ⁺¹ is
struct MPEM3{F<:Real}
    tensors :: Vector{Array{F,5}}
    z       :: Logarithmic{F}
    function MPEM3(tensors::Vector{Array{F,5}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Real}
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        all(size(At, 3) == size(At, 5) for At in tensors) ||
            throw(ArgumentError("First and third variable indices should have matching ranges because they both represent `xᵢ`"))
        return new{F}(tensors, z)
    end
end

@forward MPEM3.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex


# evaluate of MPEM3 is not simply evaluate(MatrixProductTrain{F,5}) because of how we
# interpret the entries of the tensors
function TensorTrains.evaluate(B::MPEM3, x)
    length(x) == length(B) || throw(ArgumentError("`x` must be of length $(length(B)), got $(length(x))"))
    M = [1.0;;]
    for t in 1:lastindex(B)-1
        M = M * B[t][:, :, x[t][1], x[t][2], x[t+1][1]]
    end
    M = M * B[end][:, :, x[end][1], x[end][2], 1]
    return float( only(M) / B.z)
end

# convert mpem3 into mpem2 via a Left to Right sweep of SVD's
function mpem2(B::MPEM3{F}) where {F}
    C = Vector{Array{F,4}}(undef, length(B))
    qᵢᵗ = size(B[1], 3); qⱼᵗ = size(B[1], 4); qᵢᵗ⁺¹ = size(B[1], 5)
    c = Logarithmic(one(F))

    B⁰ = B[begin]
    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= B⁰[m, n, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹]
    Bᵗ⁺¹_new = fill(1.0,1,1,1,1)  # initialize
    for t in Iterators.take(eachindex(B), length(B)-1)
        mt = maximum(abs, M)
        if !isnan(mt) && !isinf(mt) && !iszero(mt)
            M ./= mt
            c *= mt
        end
        U, λ, V = svd(M)   
        m = length(λ)     
        @cast Cᵗ[m, k, xᵢᵗ, xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k∈1:m, xᵢᵗ∈1:qᵢᵗ, xⱼᵗ∈1:qⱼᵗ
        C[t] = Cᵗ
        @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹∈1:qᵢᵗ⁺¹
        Bᵗ⁺¹ = B[t+1]
        @tullio Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] := λ[m] * 
            Vt[m, l, xᵢᵗ⁺¹] * Bᵗ⁺¹[l, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] 
        @cast M[(xᵢᵗ⁺¹, xⱼᵗ⁺¹, m), (n, xᵢᵗ⁺²)] |= Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹,xᵢᵗ⁺²]
    end
    @cast Cᵀ[m,n,xᵢ,xⱼ] := Bᵗ⁺¹_new[m,n,xᵢ,xⱼ,1]
    C[end] = Cᵀ
    return MPEM2{F}(C; z = 1/c * B.z)
end

struct PeriodicMPEM3{F<:Real}
    tensors::Vector{Array{F,5}}
    z       :: Logarithmic{F}
    function PeriodicMPEM3(tensors::Vector{Array{F,5}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Real}
        size(tensors[1],1) == size(tensors[end],2) ||
            throw(ArgumentError("Number of rows of the first matrix should coincide with the number of columns of the last matrix"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        all(size(At, 3) == size(At, 5) for At in tensors) ||
            throw(ArgumentError("First and third variable indices should have matching ranges because they both represent `xᵢ`"))
        return new{F}(tensors, z)
    end
end

@forward PeriodicMPEM3.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex

function TensorTrains.evaluate(B::PeriodicMPEM3, x)
    Tp1 = length(B)
    length(x) == Tp1 || throw(ArgumentError("`x` must be of length $(length(B)), got $(length(x))"))
    d = size(B[1], 1)
    M = Matrix(I, d, d)
    for t in eachindex(B)
        M = M * B[t][:, :, x[t][1], x[t][2], x[mod1(t+1, Tp1)][1]]
    end
    return float( tr(M) / B.z )
end

function mpem2(B::PeriodicMPEM3{F}) where {F}
    C = Vector{Array{F,4}}(undef, length(B))
    qᵢᵗ = size(B[1], 3); qⱼᵗ = size(B[1], 4); qᵢᵗ⁺¹ = size(B[1], 5)
    c = Logarithmic(one(F))

    B⁰ = B[begin]
    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= B⁰[m, n, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹]
    Bᵗ⁺¹_new = fill(1.0,1,1,1,1)  # initialize
    for t in eachindex(B)
        mt = maximum(abs, M)
        if !isnan(mt) && !isinf(mt) && !iszero(mt)
            M ./= mt
            c *= mt
        end
        U, λ, V = svd(M)   
        m = length(λ)     
        @cast Cᵗ[m, k, xᵢᵗ, xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k∈1:m, xᵢᵗ∈1:qᵢᵗ, xⱼᵗ∈1:qⱼᵗ
        C[t] = Cᵗ
        @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹∈1:qᵢᵗ⁺¹
        if t < length(B)
            Bᵗ⁺¹ = B[t+1]
            @tullio Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] := λ[m] * 
                Vt[m, l, xᵢᵗ⁺¹] * Bᵗ⁺¹[l, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] 
            @cast M[(xᵢᵗ⁺¹, xⱼᵗ⁺¹, m), (n, xᵢᵗ⁺²)] |= Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹,xᵢᵗ⁺²]
        else
            C⁰ = C[begin]
            @tullio C⁰_[m, n, xᵢ⁰, xⱼ⁰] := λ[m] * Vt[m, k, xᵢ⁰] * C⁰[k, n, xᵢ⁰, xⱼ⁰]
            C[begin] = C⁰_
        end
    end
    return PeriodicMPEM2{F}(C; z = 1/c * B.z)
end


mpem3from2(::Type{MPEM2{F}}) where F = MPEM3
mpem3from2(::Type{PeriodicMPEM2{F}}) where F = PeriodicMPEM3