MPEM1(tensors::Vector{Array{Float64,3}}) = MatrixProductTrain(tensors)

# can evaluate a MPEM1 with a vector of integers instead of a vector whose elements are 
#  1-element vectors of integers as expected by the MatrixProductTrain interface
function evaluate(A::MPEM1, x::Vector{U}) where {U<:Integer}
    tr(prod(@view a[:, :, xx...] for (a,xx) in zip(A, x)))
end

# construct a uniform mpem with given bond dimensions
mpem1(q::Int, T::Int; d::Int=2, bondsizes=fill(d, T+2)) = mpem(bondsizes, q)

# construct a uniform mpem with given bond dimensions
rand_mpem1(q::Int, T::Int; d::Int=2, bondsizes=fill(d, T+2)) = rand_mpem(bondsizes, q)

nstates(A::MPEM1) = size(A[1],3)


# at each time t, return p(x)
function marginals(A::MPEM1)
    L = accumulate_L(A)
    R = accumulate_R(A)

    A⁰ = A[begin]; R¹ = R[2]
    @reduce p⁰[x] := sum(a⁰,a¹) A⁰[a⁰,a¹,x] * R¹[a¹,a⁰]
    p⁰ ./= sum(p⁰)

    Aᵀ = A[end]; Lᵀ⁻¹ = L[end-1]
    @reduce pᵀ[x] := sum(aᵀ,a⁰) Lᵀ⁻¹[a⁰,aᵀ] * Aᵀ[aᵀ,a⁰,x]
    pᵀ ./= sum(pᵀ)

    p = map(2:getT(A)) do t 
        Lᵗ⁻¹ = L[t-1]
        Aᵗ = A[t]
        Rᵗ⁺¹ = R[t+1]
        @reduce pᵗ[x] := sum(a⁰,aᵗ,aᵗ⁺¹) Lᵗ⁻¹[a⁰,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * Rᵗ⁺¹[aᵗ⁺¹,a⁰]  
        pᵗ ./= sum(pᵗ)
    end

    return append!([p⁰], p, [pᵀ])
end


function marginals_tu(A::MPEM1)
    L = accumulate_L(A); R = accumulate_R(A); M = accumulate_M(A)
    q = size(A[1], 3)
    T = getT(A)
    b = [zeros(q, q) for _ in 0:T, _ in 0:T]
    for t in 1:T
        Lᵗ⁻¹ = t == 1 ? Matrix(1.0*I, size(A[begin],1), size(A[begin],1)) : L[t-1]
        Aᵗ = A[t]
        for u in t+1:T+1
            Rᵘ⁺¹ = u == T + 1 ? Matrix(1.0*I, size(A[end],2), size(A[end],2)) : R[u+1]
            Aᵘ = A[u]
            Mᵗᵘ = M[t, u]
            @tullio bᵗᵘ[xᵗ, xᵘ] :=
                Lᵗ⁻¹[a⁰,aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * Rᵘ⁺¹[aᵘ⁺¹,a⁰]
            b[t, u] .= bᵗᵘ ./ sum(bᵗᵘ)
        end
    end
    b
end

