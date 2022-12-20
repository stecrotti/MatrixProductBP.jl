MPEM1(tensors::Vector{Array{Float64,3}}) = MatrixProductTrain(tensors)


# construct a uniform mpem with given bond dimensions
mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = mpem(bondsizes, q)

# construct a uniform mpem with given bond dimensions
rand_mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_mpem(bondsizes, q)

nstates(A::MPEM1) = size(A[1],3)


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

