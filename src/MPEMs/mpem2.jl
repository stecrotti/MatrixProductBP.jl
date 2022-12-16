# construct a uniform mpem with given bond dimensions

MPEM2(tensors::Vector{Array{Float64, 4}}) = MatrixProductTrain(tensors)

# construct a uniform mpem with given bond dimensions
mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = mpem(T, d, bondsizes, q, q)

# construct a uniform mpem with given bond dimensions
rand_mpem2(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_mpem(T, d, bondsizes, q, q)


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

function marginalize(A::MPEM2)
    MPEM1([@tullio b[m,n,xi] := a[m,n,xi,xj] for a in A])
end
