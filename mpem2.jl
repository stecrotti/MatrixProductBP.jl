import LinearAlgebra: norm

# Matrix [Aᵗᵢⱼ(xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 4-array A[m,n,xᵢᵗ,xⱼᵗ]
# T is the final time
struct MPEM2{q,T,F<:Real} <: MPEM
    tensors :: Vector{Array{F,4}}     # Vector of length T+1
    function MPEM2(tensors::Vector{Array{F,4}}) where {F<:Real}
        T = length(tensors)-1
        q = size(tensors[1],3)
        @assert all(size(t,3) .== size(t,4) .== q for t in tensors)
        @assert size(tensors[1],1) == size(tensors[end],2) == 1
        @assert check_bond_dims2(tensors)
        new{q,T,F}(tensors)
    end
end

# construct a uniform mpem with given bond dimensions
function mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    @assert bondsizes[1] == bondsizes[end] == 1
    tensors = [ ones(bondsizes[t], bondsizes[t+1], q, q) for t in 1:T+1]
    return MPEM2(tensors)
end

# construct a uniform mpem with given bond dimensions
function rand_mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1])
    @assert bondsizes[1] == bondsizes[end] == 1
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

function bond_dims(A::MPEM2{q,T,F}) where {q,T,F}
    return [size(A[t], 2) for t in 1:lastindex(A)-1]
end

@forward MPEM2.tensors Base.getindex, Base.iterate, Base.firstindex, 
    Base.lastindex, Base.setindex!, Base.length,
    check_bond_dims2

getq(::MPEM2{q,T,F}) where {q,T,F} = q
getT(::MPEM2{q,T,F}) where {q,T,F} = T
Base.eltype(::MPEM2{q,T,F}) where {q,T,F} = F

function evaluate(A::MPEM2{q,T,F}, x) where {q,T,F}
    @assert length(x) == T + 1
    @assert all(xx[1] ∈ 1:q && xx[2] ∈ 1:q for xx in x)
    M = [1.0;;]
    for (t,Aᵗ) in enumerate(A)
        M = M * Aᵗ[:, :, x[t][1], x[t][2]]
    end
    return only(M)
end

# when truncating it assumes that matrices are already left-orthogonal
function sweep_RtoL!(C::MPEM2{q,T,F}; ε=1e-6) where {q,T,F}
    @assert ε ≤ 1
    Cᵀ = C[end]
    @cast M[m, (n, xᵢ, xⱼ)] := Cᵀ[m, n, xᵢ, xⱼ]
    Cᵗ⁻¹_trunc = rand(1,1,1,1)  # initialize

    for t in T+1:-1:2
        U, λ, V = svd(M)
        λ_norm = norm(λ)
        mprime = findlast(λₖ > ε*λ_norm for λₖ in λ)
        @assert mprime !== nothing "λ=$λ, M=$M"
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  
        # M_trunc = U_trunc * Diagonal(λ_trunc) * V_trunc'

        # X = norm(M - M_trunc)^2
        # Y = sum(abs2, λ[mprime+1:end])
        
        @cast Aᵗ[m, n, xᵢ, xⱼ] := V_trunc'[m, (n, xᵢ, xⱼ)] m in 1:mprime, xᵢ in 1:q, xⱼ in 1:q
        C[t] = Aᵗ
        
        Cᵗ⁻¹ = C[t-1]
        # @reduce Cᵗ⁻¹_trunc[m, n, xᵢ, xⱼ] := sum(k) Cᵗ⁻¹[m, k, xᵢ, xⱼ] * 
        #     U_trunc[k, n] * λ_trunc[n]
        @tullio Cᵗ⁻¹_trunc[m, n, xᵢ, xⱼ] := Cᵗ⁻¹[m, k, xᵢ, xⱼ] * 
            U_trunc[k, n] * λ_trunc[n]
        @cast M[m, (n, xᵢ, xⱼ)] := Cᵗ⁻¹_trunc[m, n, xᵢ, xⱼ]
    end
    C[begin] = Cᵗ⁻¹_trunc
    @assert check_bond_dims2(C)
    return C
end

# when truncating it assumes that matrices are already right-orthogonal
function sweep_LtoR!(C::MPEM2{q,T,F}; ε=1e-6) where {q,T,F}
    @assert ε ≤ 1
    C⁰ = C[begin]
    @cast M[(m, xᵢ, xⱼ), n] |= C⁰[m, n, xᵢ, xⱼ]
    Cᵗ⁺¹_trunc = rand(1,1,1,1)  # initialize

    for t in 1:T
        U, λ, V = svd(M)
        λ_norm = norm(λ)
        mprime = findlast(λₖ > ε*λ_norm for λₖ in λ)
        U_trunc = U[:,1:mprime]; λ_trunc = λ[1:mprime]; V_trunc = V[:,1:mprime]  
        M_trunc = U_trunc * Diagonal(λ_trunc) * V_trunc'

        X = norm(M - M_trunc)^2
        Y = sum(abs2, λ[mprime+1:end])
        @assert isapprox(X, Y, atol=1e-8) "$X, $Y"
        
        @cast Aᵗ[m, n, xᵢ, xⱼ] := U_trunc[(m, xᵢ, xⱼ), n] n in 1:mprime, xᵢ in 1:q, xⱼ in 1:q
        C[t] = Aᵗ

        Cᵗ⁺¹ = C[t+1]
        @reduce Cᵗ⁺¹_trunc[m, n, xᵢ, xⱼ] |= sum(l) λ_trunc[m] * V_trunc'[m, l] * Cᵗ⁺¹[l, n, xᵢ, xⱼ]
        @cast M[(m, xᵢ, xⱼ), n] |= Cᵗ⁺¹_trunc[m, n, xᵢ, xⱼ]
    end
    C[end] = Cᵗ⁺¹_trunc
    @assert check_bond_dims2(C)
    return C
end


function norm(A::MPEM2{q,T,F}) where {q,T,F}
    A⁰ = A[begin]
    @reduce E¹[a¹,b¹] := sum(xᵢ⁰,xⱼ⁰) A⁰[1,a¹,xᵢ⁰,xⱼ⁰] * A⁰[1,b¹,xᵢ⁰,xⱼ⁰]
    
    Eᵗ⁺¹ = E¹
    for t in 1:T
        Aᵗ = A[t+1]
        @reduce Eᵗ⁺¹[aᵗ⁺¹,bᵗ⁺¹] |= sum(xᵢᵗ,xⱼᵗ,aᵗ,bᵗ) Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Aᵗ[bᵗ,bᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Eᵗ⁺¹[aᵗ,bᵗ]
    end
    Eᵀ⁺¹ = Eᵗ⁺¹
    return sum(Eᵀ⁺¹) |> sqrt
end

# Compute norm efficiently exploiting the fact that A is left-orthonormalized
function norm_fast_L(A::MPEM2{q,T,F}) where {q,T,F}
    Aᵀ = A[end]
    @reduce N2 := sum(xᵢᵀ, xⱼᵀ, m) Aᵀ[m, 1, xᵢᵀ, xⱼᵀ] * Aᵀ[m, 1, xᵢᵀ, xⱼᵀ] 
    @assert N2 ≈ norm(A)^2 "N2=$N2, norm=$(norm(A)^2)"
    return sqrt( N2 )
end

# Compute norm efficiently exploiting the fact that A is right-orthonormalized
function norm_fast_R(A::MPEM2{q,T,F}) where {q,T,F}
    A⁰ = A[begin]
    @reduce N2 := sum(xᵢᵀ, xⱼᵀ, n) A⁰[1, n, xᵢᵀ, xⱼᵀ] * A⁰[1, n, xᵢᵀ, xⱼᵀ] 
    @assert N2 ≈ norm(A)^2 "N2=$N2, norm=$(norm(A)^2)"
    return sqrt( N2 )
end


function accumulate_L(A::MPEM2{q,T,F}) where {q,T,F}
    L = [zeros(0) for t in 0:T]
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

function accumulate_R(A::MPEM2{q,T,F}) where {q,T,F}
    R = [zeros(0) for t in 0:T]
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

# at each time t, return p(xᵢᵗ, xⱼᵗ)
function pair_marginals(A::MPEM2{q,T,F}) where {q,T,F}
    L = accumulate_L(A)
    R = accumulate_R(A)

    A⁰ = A[begin]; R¹ = R[2]
    @reduce p⁰[xᵢ⁰,xⱼ⁰] := sum(a¹) A⁰[1,a¹,xᵢ⁰,xⱼ⁰] * R¹[a¹]
    p⁰ ./= sum(p⁰)

    Aᵀ = A[end]; Lᵀ⁻¹ = L[end-1]
    @reduce pᵀ[xᵢᵀ,xⱼᵀ] := sum(aᵀ) Lᵀ⁻¹[aᵀ] * Aᵀ[aᵀ,1,xᵢᵀ,xⱼᵀ]
    pᵀ ./= sum(pᵀ)

    p = map(2:T) do t 
        Lᵗ⁻¹ = L[t-1]
        Aᵗ = A[t]
        Rᵗ⁺¹ = R[t+1]
        @reduce pᵗ[xᵢᵗ,xⱼᵗ] := sum(aᵗ,aᵗ⁺¹) Lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Rᵗ⁺¹[aᵗ⁺¹]  
        pᵗ ./= sum(pᵗ)
    end

    return [p⁰, p..., pᵀ]
end

function firstvar_marginals(A::MPEM2{q,T,F}; p = pair_marginals(A)) where {q,T,F}
    map(p) do pₜ
        pᵢᵗ =  sum(pₜ, dims=2) |> vec
        pᵢᵗ ./= sum(pᵢᵗ)
    end
end