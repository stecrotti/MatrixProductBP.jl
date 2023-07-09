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

isapprox(A::T, B::T; kw...) where {T<:MatrixProductTrain} = isapprox(A.tensors, B.tensors; kw...)

const MPEM1{F} = MatrixProductTrain{F, 3}
const MPEM2{F} = MatrixProductTrain{F, 4}
# const MPEM3{F} = MatrixProductTrain{F, 5}

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



# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)  # initialize

    for t in getT(C)+1:-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])
    return C
end

# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(C::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    D = fill(1.0,1,1,1)  # initialize

    for t in 1:getT(C)
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    C[end] = _reshapeas(D, C[end])
    return C
end

function compress!(A::MatrixProductTrain; svd_trunc=TruncThresh(1e-6))
    sweep_LtoR!(A, svd_trunc=TruncThresh(0.0))
    sweep_RtoL!(A; svd_trunc)
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
            @tullio Mᵗᵘ[aᵗ⁺¹, aᵘ] := Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹] * Aᵘ⁻¹[aᵘ⁻¹, aᵘ, x]
            M[t, u] = Mᵗᵘ
            Mᵗᵘ⁻¹, Mᵗᵘ = Mᵗᵘ, Mᵗᵘ⁻¹
        end
    end

    return M
end

# p(xᵗ) for each t
function marginals(A::MatrixProductTrain{F,N};
        L = accumulate_L(A), R = accumulate_R(A)) where {F,N}
    
    A⁰ = _reshape1(A[begin]); R¹ = R[2]
    @reduce p⁰[x] := sum(a¹) A⁰[1,a¹,x] * R¹[a¹]
    p⁰ ./= sum(p⁰)
    p⁰ = reshape(p⁰, size(A[begin])[3:end])

    Aᵀ = _reshape1(A[end]); Lᵀ⁻¹ = L[end-1]
    @reduce pᵀ[x] := sum(aᵀ) Lᵀ⁻¹[aᵀ] * Aᵀ[aᵀ,1,x]
    pᵀ ./= sum(pᵀ)
    pᵀ = reshape(pᵀ, size(A[end])[3:end])

    p = map(2:getT(A)) do t 
        Lᵗ⁻¹ = L[t-1]
        Aᵗ = _reshape1(A[t])
        Rᵗ⁺¹ = R[t+1]
        @reduce pᵗ[x] := sum(aᵗ,aᵗ⁺¹) Lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ⁺¹[aᵗ⁺¹]  
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end

    return append!([p⁰], p, [pᵀ])
end

# p(xᵗ,xᵘ) for all (t,u)
function marginals_tu(A::MatrixProductTrain{F,N};
        L = accumulate_L(A), R = accumulate_R(A), M = accumulate_M(A),
        Δtmax = getT(A)) where {F,N}
    T = getT(A)
    qs = tuple(reduce(vcat, [x,x] for x in size(A[begin])[3:end])...)
    b = Array{F,2*(N-2)}[zeros(ones(Int, 2*(N-2))...) for _ in 0:T, _ in 0:T]
    for t in 1:T
        Lᵗ⁻¹ = t == 1 ? [1.0;] : L[t-1]
        Aᵗ = _reshape1(A[t])
        for u in t+1:min(T+1,t+Δtmax)
            Rᵘ⁺¹ = u == T + 1 ? [1.0;] : R[u+1]
            Aᵘ = _reshape1(A[u])
            Mᵗᵘ = M[t, u]
            @tullio bᵗᵘ[xᵗ, xᵘ] :=
                Lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * Rᵘ⁺¹[aᵘ⁺¹]
            bᵗᵘ ./= sum(bᵗᵘ)
            b[t,u] = reshape(bᵗᵘ, qs)
        end
    end
    b
end

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

# return a new MPTrain such that `A(x)+B(x)=(A+B)(x)`. Matrix sizes are doubled
+(A::MatrixProductTrain, B::MatrixProductTrain) = _compose(+, A, B)
-(A::MatrixProductTrain, B::MatrixProductTrain) = _compose(-, A, B)

function _compose(f, A::MatrixProductTrain{F,NA}, B::MatrixProductTrain{F,NB}) where {F,NA,NB}
    @assert NA == NB
    @assert length(A) == length(B)
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ hcat(Aᵗ[:,:,x...], f(Bᵗ[:,:,x...])) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), 1, sa[2]+sb[2], size(Aᵗ)[3:end]...)
        elseif t == lastindex(A)
            Cᵗ = [ vcat(Aᵗ[:,:,x...], Bᵗ[:,:,x...]) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), sa[1]+sb[1], 1, size(Aᵗ)[3:end]...)
        else
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bᵗ[:,:,x...]] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        end
    end
    MatrixProductTrain(tensors)
end

# hierarchical sampling p(x) = p(x⁰)p(x¹|x⁰)p(x²|x¹,x⁰) ...
# returns `x,p`, the sampled sequence and its probability
function sample!(rng::AbstractRNG, x, A::MatrixProductTrain{F,N};
        R = accumulate_R(A)) where {F,N}
    T = getT(A)
    @assert length(x) == T + 1
    @assert all(length(xᵗ) == N-2 for xᵗ in x)

    Q = ones(F, 1)  # stores product of the first `t` matrices, evaluated at the sampled `x⁰,x¹,...,xᵗ`
    for t in eachindex(A)
        Rᵗ⁺¹ = t == T+1 ? ones(F,1) : R[t+1]
        # collapse multivariate xᵗ into 1D vector, sample from it
        Aᵗ = _reshape1(A[t])
        @tullio p[x] := Q[m] * Aᵗ[m,n,x] * Rᵗ⁺¹[n]
        p ./= sum(p)
        xᵗ = sample_noalloc(rng, p)
        x[t] .= CartesianIndices(size(A[t])[3:end])[xᵗ] |> Tuple
        # update prob
        Q = Q * Aᵗ[:,:,xᵗ]
    end
    p = only(Q) / only(first(R))
    return x, p
end

function sample!(x, A::MatrixProductTrain{F,N}; R = accumulate_R(A)) where {F,N}
    sample!(GLOBAL_RNG, x, A; R)
end
function sample(rng::AbstractRNG, A::MatrixProductTrain{F,N};
        R = accumulate_R(A)) where {F,N}
    x = [zeros(Int, N-2) for Aᵗ in A]
    sample!(rng, x, A; R)
end
function sample(A::MatrixProductTrain{F,N}; R = accumulate_R(A)) where {F,N}
    sample(GLOBAL_RNG, A; R)
end