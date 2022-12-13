# Matrix [Aᵗᵢⱼ(xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 4-array A[m,n,xᵢᵗ,xⱼᵗ]
struct MPEM2{F} <: MPEM
    tensors :: Vector{Array{F,4}}     # Vector of length T+1

    function MPEM2(tensors::Vector{Array{F,4}}) where {F}
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims2(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F}(tensors)
    end
end

# construct a uniform mpem with given bond dimensions
function mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    bondsizes[1] == bondsizes[end] == 1 || 
        throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
    tensors = [ ones(bondsizes[t], bondsizes[t+1], q, q) for t in 1:T+1]
    return MPEM2(tensors)
end

# construct a uniform mpem with given bond dimensions
function rand_mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    bondsizes[1] == bondsizes[end] == 1 || 
        throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
    tensors = [ rand(bondsizes[t], bondsizes[t+1], q, q) for t in 1:T+1]
    return MPEM2(tensors)
end

function check_bond_dims2(tensors::Vector{<:Array})
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

function bond_dims(A::MPEM2)
    return [size(A[t], 2) for t in 1:lastindex(A)-1]
end

@forward MPEM2.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    length, check_bond_dims2

getT(A::MPEM2) = length(A.tensors) - 1
eltype(::MPEM2{F}) where F = F

function evaluate(A::MPEM2, x)
    length(x) == getT(A) + 1 || throw(ArgumentError("`x` must be of length $(getT(A)+1), got $(length(x))"))
    # all(xx[1] ∈ 1:q && xx[2] ∈ 1:q for xx in x) || throw(ArgumentError("All `x`'s must be in domain 1:$q")) 
    M = [1.0;;]
    for (t,Aᵗ) in enumerate(A)
        M = M * Aᵗ[:, :, x[t][1], x[t][2]]
    end
    return only(M)
end


function MPEM1(C::MPEM2)
    MPEM1([reshape(c, size(c,1), size(c,2), size(c,3)*size(c,4)) for c in C.tensors])
end

# function MPEM2(C::MPEM1, D::MPEM2)
#     MPEM2([reshape(c, size(c,1), size(c,2), size(d,3), size(d,4)) for (c,d) in zip(C.tensors,D.tensors)])
# end


# sweep_RtoL!(C::MPEM2; svd_trunc::SVDTrunc=TruncThresh(1e-6)) = 
#         MPEM2(sweep_RtoL!(MPEM1(C); svd_trunc), C)

# sweep_LtoR!(C::MPEM2; svd_trunc::SVDTrunc=TruncThresh(1e-6)) = 
#         MPEM2(sweep_RtoL!(MPEM1(C); svd_trunc), C)

# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MPEM2; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    Cᵀ = C[end]
    qᵢ = size(Cᵀ, 3); qⱼ = size(Cᵀ, 4)
    @cast M[m, (n, xᵢ, xⱼ)] := Cᵀ[m, n, xᵢ, xⱼ]
    Cᵗ⁻¹_trunc = rand(1,1,1,1)  # initialize

    for t in getT(C)+1:-1:2
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  
        @cast Aᵗ[m, n, xᵢ, xⱼ] := V_trunc'[m, (n, xᵢ, xⱼ)] m in 1:mprime, xᵢ in 1:qᵢ, xⱼ in 1:qⱼ
        C[t] = Aᵗ
        
        Cᵗ⁻¹ = C[t-1]
        qᵢ = size(Cᵗ⁻¹, 3); qⱼ = size(Cᵗ⁻¹, 4)
        @tullio Cᵗ⁻¹_trunc[m, n, xᵢ, xⱼ] := Cᵗ⁻¹[m, k, xᵢ, xⱼ] * 
            U_trunc[k, n] * λ_trunc[n]
        @cast M[m, (n, xᵢ, xⱼ)] := Cᵗ⁻¹_trunc[m, n, xᵢ, xⱼ]
    end
    C[begin] = Cᵗ⁻¹_trunc
    @assert check_bond_dims2(C)
    return C
end

# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(C::MPEM2; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    C⁰ = C[begin]
    qᵢ = size(C⁰, 3); qⱼ = size(C⁰, 4)
    @cast M[(m, xᵢ, xⱼ), n] |= C⁰[m, n, xᵢ, xⱼ]
    Cᵗ⁺¹_trunc = rand(1,1,1,1)  # initialize

    for t in 1:getT(C)
        U, λ, V = svd(M)
        mprime = svd_trunc(λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  

        @cast Aᵗ[m, n, xᵢ, xⱼ] := U_trunc[(m, xᵢ, xⱼ), n] n in 1:mprime, xᵢ in 1:qᵢ, xⱼ in 1:qⱼ
        C[t] = Aᵗ

        Cᵗ⁺¹ = C[t+1]
        qᵢ = size(Cᵗ⁺¹, 3); qⱼ = size(Cᵗ⁺¹, 4)
        @tullio Cᵗ⁺¹_trunc[m, n, xᵢ, xⱼ] := λ_trunc[m] * V_trunc'[m, l] * 
            Cᵗ⁺¹[l, n, xᵢ, xⱼ]
        @cast M[(m, xᵢ, xⱼ), n] |= Cᵗ⁺¹_trunc[m, n, xᵢ, xⱼ]
    end
    C[end] = Cᵗ⁺¹_trunc
    @assert check_bond_dims2(C)
    return C
end

function accumulate_L(A::MPEM2)
    T = getT(A)
    L = [zeros(0) for _ in 0:T]
    A⁰ = A[begin]
    @reduce L⁰[a¹] := sum(xᵢ⁰,xⱼ⁰) A⁰[1,a¹,xᵢ⁰,xⱼ⁰]
    L[1] = L⁰

    Lᵗ = L⁰
    for t in 1:T
        Aᵗ = A[t+1]
        @reduce Lᵗ[aᵗ⁺¹] |= sum(xᵢᵗ,xⱼᵗ,aᵗ) Lᵗ[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] 
        L[t+1] = Lᵗ
    end
    return L
end

function accumulate_R(A::MPEM2)
    T = getT(A)
    R = [zeros(0) for _ in 0:T]
    Aᵀ = A[end]
    @reduce Rᵀ[aᵀ] := sum(xᵢᵀ,xⱼᵀ) Aᵀ[aᵀ,1,xᵢᵀ,xⱼᵀ]
    R[end] = Rᵀ

    Rᵗ = Rᵀ
    for t in T:-1:1
        Aᵗ = A[t]
        @reduce Rᵗ[aᵗ] |= sum(xᵢᵗ,xⱼᵗ,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Rᵗ[aᵗ⁺¹] 
        R[t] = Rᵗ
    end
    return R
end

function accumulate_M(A::MPEM2)
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
            @reduce Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ] |= sum(aᵘ⁻¹, xᵢᵘ⁻¹, xⱼᵘ⁻¹) Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹] * Aᵘ⁻¹[aᵘ⁻¹, aᵘ, xᵢᵘ⁻¹, xⱼᵘ⁻¹]
            M[t, u] = Mᵗᵘ⁻¹
        end
    end

    return M
end

# at each time t, return p(xᵢᵗ, xⱼᵗ)
function pair_marginal(A::MPEM2)
    L = accumulate_L(A)
    R = accumulate_R(A)

    A⁰ = A[begin]; R¹ = R[2]
    @reduce p⁰[xᵢ⁰,xⱼ⁰] := sum(a¹) A⁰[1,a¹,xᵢ⁰,xⱼ⁰] * R¹[a¹]
    p⁰ ./= sum(p⁰)

    Aᵀ = A[end]; Lᵀ⁻¹ = L[end-1]
    @reduce pᵀ[xᵢᵀ,xⱼᵀ] := sum(aᵀ) Lᵀ⁻¹[aᵀ] * Aᵀ[aᵀ,1,xᵢᵀ,xⱼᵀ]
    pᵀ ./= sum(pᵀ)

    p = map(2:getT(A)) do t 
        Lᵗ⁻¹ = L[t-1]
        Aᵗ = A[t]
        Rᵗ⁺¹ = R[t+1]
        @reduce pᵗ[xᵢᵗ,xⱼᵗ] := sum(aᵗ,aᵗ⁺¹) Lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Rᵗ⁺¹[aᵗ⁺¹]  
        pᵗ ./= sum(pᵗ)
    end

    return [p⁰, p..., pᵀ]
end

function firstvar_marginal(A::MPEM2; p = pair_marginal(A))
    map(p) do pₜ
        pᵢᵗ =  sum(pₜ, dims=2) |> vec
        pᵢᵗ ./= sum(pᵢᵗ)
    end
end

function pair_marginal_tu(A::MPEM2; showprogress::Bool=true)
    l = accumulate_L(A); r = accumulate_R(A); m = accumulate_M(A)
    q = size(A[1], 3)
    T = getT(A)
    b = [zeros(q, q, q, q) for _ in 0:T, _ in 0:T]
    for t in 1:T
        lᵗ⁻¹ = t == 1 ? [1.0;] : l[t-1]
        Aᵗ = A[t]
        for u in t+1:T+1
            rᵘ⁺¹ = u == T + 1 ? [1.0;] : r[u+1]
            Aᵘ = A[u]
            mᵗᵘ = m[t, u]
            @tullio bᵗᵘ[xᵢᵗ, xⱼᵗ, xᵢᵘ, xⱼᵘ] :=
                lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵢᵗ, xⱼᵗ] * mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵢᵘ, xⱼᵘ] * rᵘ⁺¹[aᵘ⁺¹]
            b[t, u] .= bᵗᵘ ./ sum(bᵗᵘ)
        end
    end
    b
end

function firstvar_marginal_tu(A::MPEM2; showprogress::Bool=true, 
        p_tu = pair_marginal_tu(A; showprogress))
    map(p_tu) do pᵗᵘ
        pᵢᵗᵘ =  sum(sum(pᵗᵘ, dims=2), dims=4)[:,1,:,1]
    end
end

# compute normalization of an MPEM2 efficiently
function normalization(A::MPEM2; l = accumulate_L(A), r = accumulate_R(A))
    z = only(l[end])
    @assert only(r[begin]) ≈ z "z=$z, got $(only(r[begin])), A=$A"  # sanity check
    z
end

# normalize so that the sum over all pair trajectories is 1.
# return log of the normalization
function normalize!(A::MPEM2)
    c = normalize_eachmatrix!(A)
    Z = normalization(A)
    T = getT(A)
    for a in A
        a ./= Z^(1/(T+1))
    end
    c + log(Z)
end

function marginalize(A::MPEM2)
    MPEM1([@tullio b[m,n,xi] := a[m,n,xi,xj] for a in A])
end