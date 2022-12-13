# Matrix [Aᵗᵢⱼ(xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 4-array A[m,n,xᵢᵗ,xⱼᵗ]
struct MPEM1{F<:Real} <: MPEM
    tensors :: Vector{Array{F,3}}     # Vector of length T+1

    function MPEM1(tensors::Vector{Array{F,3}}) where {F<:Real}
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims2(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F}(tensors)
    end
end

# construct a uniform mpem with given bond dimensions
function mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    bondsizes[1] == bondsizes[end] == 1 || 
        throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
    tensors = [ ones(bondsizes[t], bondsizes[t+1], q) for t in 1:T+1]
    return MPEM1(tensors)
end

# construct a uniform mpem with given bond dimensions
function rand_mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    bondsizes[1] == bondsizes[end] == 1 || 
        throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
    tensors = [ rand(bondsizes[t], bondsizes[t+1], q) for t in 1:T+1]
    return MPEM1(tensors)
end

function check_bond_dims1(tensors::Vector{<:Array})
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

function bond_dims(A::MPEM1)
    return [size(A[t], 2) for t in 1:lastindex(A)-1]
end

@forward MPEM1.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    length, check_bond_dims1

getT(A::MPEM1) = length(A) - 1
eltype(::MPEM1{F}) where F = F

evaluate(A::MPEM1, x) = only(prod(@view Aᵗ[:, :, xᵗ] for (xᵗ,Aᵗ) in zip(x,A)))


# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MPEM1; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    Cᵀ = C[end]
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    Cᵗ⁻¹_trunc = fill(1.0,1,1,1)  # initialize

    for t in getT(C)+1:-1:2
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]
        @cast Aᵗ[m, n, x] := V_trunc'[m, (n, x)] m in 1:mprime, x in 1:q
        C[t] = Aᵗ
        
        Cᵗ⁻¹ = C[t-1]
        @tullio Cᵗ⁻¹_trunc[m, n, x] := Cᵗ⁻¹[m, k, x] * U_trunc[k, n] * λ_trunc[n]
        @cast M[m, (n, x)] := Cᵗ⁻¹_trunc[m, n, x]
    end
    C[begin] = Cᵗ⁻¹_trunc
    @assert check_bond_dims1(C.tensors)
    return C
end

# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(C::MPEM1; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    C⁰ = C[begin]
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    Cᵗ⁺¹_trunc = fill(1.0,1,1,1)  # initialize

    for t in 1:getT(C)
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  

        @cast Aᵗ[m, n, x] := U_trunc[(m, x), n] n in 1:mprime, x in 1:q
        C[t] = Aᵗ
        Cᵗ⁺¹ = C[t+1]
        @tullio Cᵗ⁺¹_trunc[m, n, x] := λ_trunc[m] * V_trunc'[m, l] * Cᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= Cᵗ⁺¹_trunc[m, n, x]
    end
    C[end] = Cᵗ⁺¹_trunc
    @assert check_bond_dims1(C)
    return C
end

function accumulate_L(A::MPEM1)
    T = getT(A)
    L = [zeros(0) for _ in 0:T]
    A⁰ = A[begin]
    @reduce L⁰[a¹] := sum(x) A⁰[1,a¹,x]
    L[1] = L⁰

    Lᵗ = L⁰
    for t in 1:T
        Aᵗ = A[t+1]
        @reduce Lᵗ[aᵗ⁺¹] |= sum(x,aᵗ) Lᵗ[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] 
        L[t+1] = Lᵗ
    end
    return L
end

function accumulate_R(A::MPEM1)
    T = getT(A)
    R = [zeros(0) for _ in 0:T]
    Aᵀ = A[end]
    @reduce Rᵀ[aᵀ] := sum(x) Aᵀ[aᵀ,1,x]
    R[end] = Rᵀ

    Rᵗ = Rᵀ
    for t in T:-1:1
        Aᵗ = A[t]
        @reduce Rᵗ[aᵗ] |= sum(x,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ[aᵗ⁺¹] 
        R[t] = Rᵗ
    end
    return R
end

function accumulate_M(A::MPEM1)
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
            Aᵘ⁻¹ = A[u-1]
            @reduce Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ] |= sum(aᵘ⁻¹, x) Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹] * Aᵘ⁻¹[aᵘ⁻¹, aᵘ, x]
            M[t, u] = Mᵗᵘ⁻¹
        end
    end

    return M
end

# at each time t, return p(x)
function marginals(A::MPEM1)
    L = accumulate_L(A)
    R = accumulate_R(A)

    A⁰ = A[begin]; R¹ = R[2]
    @reduce p⁰[x] := sum(a¹) A⁰[1,a¹,x] * R¹[a¹]
    p⁰ ./= sum(p⁰)

    Aᵀ = A[end]; Lᵀ⁻¹ = L[end-1]
    @reduce pᵀ[x] := sum(aᵀ) Lᵀ⁻¹[aᵀ] * Aᵀ[aᵀ,1,x]
    pᵀ ./= sum(pᵀ)

    p = map(2:getT(A)) do t 
        Lᵗ⁻¹ = L[t-1]
        Aᵗ = A[t]
        Rᵗ⁺¹ = R[t+1]
        @reduce pᵗ[x] := sum(aᵗ,aᵗ⁺¹) Lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ⁺¹[aᵗ⁺¹]  
        pᵗ ./= sum(pᵗ)
    end

    return append!([p⁰], p, [pᵀ])
end


function marginals_tu(A::MPEM1; showprogress::Bool=true)
    l = accumulate_L(A); r = accumulate_R(A); m = accumulate_M(A)
    q = size(A[1], 3)
    T = getT(A)
    b = [zeros(q, q) for _ in 0:T, _ in 0:T]
    for t in 1:T
        lᵗ⁻¹ = t == 1 ? [1.0;] : l[t-1]
        Aᵗ = A[t]
        for u in t+1:T+1
            rᵘ⁺¹ = u == T + 1 ? [1.0;] : r[u+1]
            Aᵘ = A[u]
            mᵗᵘ = m[t, u]
            @tullio bᵗᵘ[xᵗ, xᵘ] :=
                lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * rᵘ⁺¹[aᵘ⁺¹]
            b[t, u] .= bᵗᵘ ./ sum(bᵗᵘ)
        end
    end
    b
end

# compute normalization of an MPEM1 efficiently
function normalization(A::MPEM1; l = accumulate_L(A), r = accumulate_R(A))
    z = only(l[end])
    @assert only(r[begin]) ≈ z "z=$z, got $(only(r[begin])), A=$A"  # sanity check
    z
end

# normalize so that the sum over all pair trajectories is 1.
# return log of the normalization
function normalize!(A::MPEM1)
    c = normalize_eachmatrix!(A)
    Z = normalization(A)
    T = getT(A)
    for a in A
        a ./= Z^(1/(T+1))
    end
    c + log(Z)
end