"""
[Aᵢⱼ] ⨉ 🚂
"""
struct MatrixProductTrain{F<:Real, N} <: MPEM
    tensors::Vector{Array{F,N}}
    function MatrixProductTrain(tensors::Vector{Array{F,N}}) where {F<:Real, N}
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F,N}(tensors)
    end
end

@forward MatrixProductTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex


function check_bond_dims(tensors::Vector{<:Array})
    for t in eachindex(tensors)
        dᵗ = size(tensors[t],2)
        if t == lastindex(tensors)
            dᵗ⁺¹ = size(tensors[1],1)
        else
            dᵗ⁺¹ = size(tensors[t+1],1)
        end
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end

# keep size of matrix elements under control by dividing by the max
# return the log of the product of the individual normalizations 
function normalize_eachmatrix!(A::MatrixProductTrain)
    c = 0.0
    for m in A
        mm = maximum(abs, m)
        if !any(isnan, mm) && !any(isinf, mm)
            m ./= mm
            c += log(mm)
        end
    end
    c
end

isapprox(A::T, B::T; kw...) where {T<:MatrixProductTrain} = isapprox(A.tensors, B.tensors; kw...)

const MPEM1{F} = MatrixProductTrain{F, 3}
const MPEM2{F} = MatrixProductTrain{F, 4}

"Construct a uniform mpem with given bond dimensions"
mpem(bondsizes, q...) = MatrixProductTrain([ones(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])

"Construct a random mpem with given bond dimensions"
rand_mpem(bondsizes, q...) = MatrixProductTrain([rand(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])

bond_dims(A::MPEM) = [size(A[t], 1) for t in eachindex(A)]

getT(A::MatrixProductTrain) = length(A) - 1

eltype(::MatrixProductTrain{F,N}) where {N,F} = F

evaluate(A::MatrixProductTrain, X...) = tr(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))

_reshape1(x) = reshape(x, size(x,1), size(x,2), prod(size(x)[3:end])...)
_reshapeas(x,y) = reshape(x, size(x,1), size(x,2), size(y)[3:end]...)


# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[m, (n, x)] := C⁰[m, n, x]
    U, λ, V = svd_trunc(M)
    @cast A⁰[m, n, x] := V'[m, (n, x)] x in 1:q
    C[begin] = _reshapeas(A⁰, C[begin])     
    Cᵗ⁻¹ = _reshape1(C[end])
    @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
    @cast M[m, (n, x)] := D[m, n, x]

    for t in getT(C)+1:-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])

    @assert check_bond_dims(C.tensors)

    return C
end

# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(A::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    A⁰ = _reshape1(A[begin])
    q = size(A⁰, 3)
    @cast M[(m, x), n] |= A⁰[m, n, x]
    D = fill(1.0,1,1,1)  # initialize

    for t in 1:getT(A)
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x in 1:q
        A[t] = _reshapeas(Aᵗ, A[t])
        Aᵗ⁺¹ = _reshape1(A[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Aᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    U, λ, V = svd_trunc(M)
    @cast Aᵀ[m, n, x] := U[(m, x), n] x in 1:q
    A[end] = _reshapeas(Aᵀ, A[end])
    A⁰ = _reshape1(A[begin])
    @tullio D[m, n, x] := λ[m] * V'[m, l] * A⁰[l, n, x]
    A[begin] = _reshapeas(D,  A[begin])

    return A
end

function compress!(A::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    sweep_LtoR!(A, svd_trunc=TruncThresh(0.0))
    sweep_RtoL!(A; svd_trunc)
end

function accumulate_L(A::MatrixProductTrain)
    T = getT(A)
    L = [zeros(0,0) for _ in 0:T]
    A⁰ = _reshape1(A[begin])
    @reduce L⁰[a⁰,a¹] := sum(x) A⁰[a⁰,a¹,x]
    L[1] = L⁰

    Lᵗ = L⁰
    for t in 1:T
        Aᵗ = _reshape1(A[t+1])
        @reduce Lᵗ[a⁰,aᵗ⁺¹] |= sum(x,aᵗ) Lᵗ[a⁰,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] 
        L[t+1] = Lᵗ
    end
    return L
end

function accumulate_R(A::MatrixProductTrain)
    T = getT(A)
    R = [zeros(0,0) for _ in 0:T]
    Aᵀ = _reshape1(A[end])
    @reduce Rᵀ[aᵀ,a⁰] := sum(x) Aᵀ[aᵀ,a⁰,x]
    R[end] = Rᵀ

    Rᵗ = Rᵀ
    for t in T:-1:1
        Aᵗ = _reshape1(A[t])
        @reduce Rᵗ[aᵗ,a⁰] |= sum(x,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ[aᵗ⁺¹,a⁰] 
        R[t] = Rᵗ
    end
    return R
end

function accumulate_M(A::MatrixProductTrain)
    T = getT(A)
    M = [zeros(0, 0) for _ in 0:T, _ in 0:T]
    
    # initial condition
    for t in 1:T
        range_aᵗ⁺¹ = axes(A[t+1], 1)
        Mᵗᵗ⁺¹ = [float((a == c)) for a in range_aᵗ⁺¹, c in range_aᵗ⁺¹]
        M[t, t+1] = Mᵗᵗ⁺¹
    end

    for t in 1:getT(A)
        Mᵗᵘ⁻¹ = M[t, t+1]
        for u in t+2:T+1
            Aᵘ⁻¹ = _reshape1(A[u-1])
            @reduce Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ] |= sum(aᵘ⁻¹, x) Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹] * Aᵘ⁻¹[aᵘ⁻¹, aᵘ, x]
            M[t, u] = Mᵗᵘ⁻¹
        end
    end

    return M
end

# compute normalization of an MPEM1 efficiently
function normalization(A::MatrixProductTrain; L = accumulate_L(A), R = accumulate_R(A))
    Z = tr(L[end])
    @assert tr(R[begin]) ≈ Z "Z=$Z, got $(tr(r[begin])), A=$A"  # sanity check
    Z
end

# normalize so that the sum over all trajectories is 1.
# return log of the normalization
function normalize!(A::MatrixProductTrain)
    c = normalize_eachmatrix!(A)
    Z = normalization(A)
    T = getT(A)
    for a in A
        a ./= Z^(1/(T+1))
    end
    c + log(Z)
end

# return a new MPTrain such that `A(x)+B(x)=(A+B)(x)`. Matrix sizes are doubled
+(A::MatrixProductTrain, B::MatrixProductTrain) = _compose(+, A, B)
-(A::MatrixProductTrain, B::MatrixProductTrain) = _compose(-, A, B)

function _compose(f, A::MatrixProductTrain{F,NA}, B::MatrixProductTrain{F,NB}) where {F,NA,NB}
    @assert NA == NB
    @assert length(A) == length(B)
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) f(Bᵗ[:,:,x...])] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        else
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bᵗ[:,:,x...]] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        end
    end
    MatrixProductTrain(tensors)
end