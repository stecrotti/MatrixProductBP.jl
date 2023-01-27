"""
[Aᵢⱼ] ⨉ 🚂
"""
struct MatrixProductTrain{F<:Real, N} <: MPEM
    tensors::Vector{Array{F,N}}
    function MatrixProductTrain(tensors::Vector{Array{F,N}}) where {F<:Real, N}
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F,N}(tensors)
    end
end

@forward MatrixProductTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex


function check_bond_dims(tensors::Vector{<:Array})
    for t in 1:lastindex(tensors)-1
        dᵗ = size(tensors[t],2)
        dᵗ⁺¹ = size(tensors[t+1],1)
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

-(A::T, B::T) where {T<:MatrixProductTrain} = MPEM([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])

isapprox(A::T, B::T; kw...) where {T<:MatrixProductTrain} = isapprox(A.tensors, B.tensors; kw...)

const MPEM1{F} = MatrixProductTrain{F, 3}
const MPEM2{F} = MatrixProductTrain{F, 4}
const MPEM3{F} = MatrixProductTrain{F, 5}

"Construct a uniform mpem with given bond dimensions"
mpem(bondsizes, q...) = MatrixProductTrain([ones(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])

"Construct a random mpem with given bond dimensions"
rand_mpem(bondsizes, q...) = MatrixProductTrain([rand(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])

bond_dims(A::MPEM) = [size(A[t], 2) for t in 1:lastindex(A)-1]

getT(A::MatrixProductTrain) = length(A) - 1

eltype(::MatrixProductTrain{F,N}) where {N,F} = F

evaluate(A::MatrixProductTrain, X...) = only(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))

_reshape1(x) = reshape(x, size(x,1), size(x,2), prod(size(x)[3:end])...)
_reshapeas(x,y) = reshape(x, size(x,1), size(x,2), size(y)[3:end]...)

# print info about truncations
function _debug_svd(t, M, U, λ, V, mprime)
    @debug "svd" """t=$t M$(size(M))=U$(size(U))*Λ$((length(λ),length(λ)))*V$(size(V'))
    Truncation to $mprime singular values.
    Error=$(sum(abs2, λ[mprime+1:end]) / sum(abs2, λ) |> sqrt)"""
end

# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MatrixProductTrain; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    Cᵗ⁻¹_trunc = fill(1.0,1,1,1)  # initialize

    for t in getT(C)+1:-1:2
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        _debug_svd(t, M, U, λ, V, mprime)
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]
        @cast Aᵗ[m, n, x] := V_trunc'[m, (n, x)] m in 1:mprime, x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio Cᵗ⁻¹_trunc[m, n, x] := Cᵗ⁻¹[m, k, x] * U_trunc[k, n] * λ_trunc[n]
        @cast M[m, (n, x)] := Cᵗ⁻¹_trunc[m, n, x]
    end
    C[begin] = _reshapeas(Cᵗ⁻¹_trunc, C[begin])
    return C
end


# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(C::MatrixProductTrain; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    Cᵗ⁺¹_trunc = fill(1.0,1,1,1)  # initialize

    for t in 1:getT(C)
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        _debug_svd(t, M, U, λ, V, mprime)
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  
        @cast Aᵗ[m, n, x] := U_trunc[(m, x), n] n in 1:mprime, x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio Cᵗ⁺¹_trunc[m, n, x] := λ_trunc[m] * V_trunc'[m, l] * Cᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= Cᵗ⁺¹_trunc[m, n, x]
    end
    C[end] = _reshapeas(Cᵗ⁺¹_trunc, C[end])
    return C
end

function accumulate_L(A::MatrixProductTrain)
    T = getT(A)
    L = [zeros(0) for _ in 0:T]
    A⁰ = _reshape1(A[begin])
    @reduce L⁰[a¹] := sum(x) A⁰[1,a¹,x]
    L[1] = L⁰

    Lᵗ = L⁰
    for t in 1:T
        Aᵗ = _reshape1(A[t+1])
        @reduce Lᵗ[aᵗ⁺¹] |= sum(x,aᵗ) Lᵗ[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] 
        L[t+1] = Lᵗ
    end
    return L
end

function accumulate_R(A::MatrixProductTrain)
    T = getT(A)
    R = [zeros(0) for _ in 0:T]
    Aᵀ = _reshape1(A[end])
    @reduce Rᵀ[aᵀ] := sum(x) Aᵀ[aᵀ,1,x]
    R[end] = Rᵀ

    Rᵗ = Rᵀ
    for t in T:-1:1
        Aᵗ = _reshape1(A[t])
        @reduce Rᵗ[aᵗ] |= sum(x,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ[aᵗ⁺¹] 
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
function normalization(A::MatrixProductTrain; l = accumulate_L(A), r = accumulate_R(A))
    z = only(l[end])
    @assert only(r[begin]) ≈ z "z=$z, got $(only(r[begin])), A=$A"  # sanity check
    z
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
