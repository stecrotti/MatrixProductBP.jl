"""
Factor for the factor graph of a model solvable with MPBP.

Any `BPFactor` subtype must implement:
- A functor that computes the Boltzmann contribution to the joint probability
- `nstates()::Type{<:BPFactor})` returning the number of allowed states for a variable

That's it!

"""
abstract type BPFactor; end

nstates(::Type{<:BPFactor}) = error("Not implemented")

# compute outgoing message as a function of the incoming ones
# A is a vector with all incoming messages. At index j_index there is m(j → i)
# ψᵢⱼ are the ones living on the outedges of node i
function f_bp(A::Vector{MPEM2{F}}, wᵢ::Vector{U}, 
        ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}}, j_index::Integer;
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0)) where {F,U<:BPFactor}
    T = getT(A[1])
    @assert all(getT(a) == T for a in A)
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1
    q = length(ϕᵢ[1])
    @assert all(length(ϕᵢᵗ) == q for ϕᵢᵗ in ϕᵢ)
    @assert j_index in eachindex(A)
    z = length(A)      # z = |∂i|
    x_neigs = Iterators.product(fill(1:q, z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T + 1)
 
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    for t in 1:T+1
        # select incoming A's but not the j-th one
        Aᵗ = kron2([A[k][t] for k in eachindex(A)[Not(j_index)]]...)
        nrows, ncols = size(Aᵗ, 1), size(Aᵗ, 2)
        Bᵗ = zeros(nrows, ncols, q, q, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xₙᵢᵗ in x_neigs
                    xⱼᵗ = xₙᵢᵗ[j_index]
                    xₙᵢ₋ⱼᵗ = xₙᵢᵗ[Not(j_index)]
                    Bᵗ[:, :, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹] .+= (t == T + 1 ? 1.0 : wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) *
                        Aᵗ[:, :, xᵢᵗ, xₙᵢ₋ⱼᵗ...] *
                        prod(ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in enumerate(xₙᵢᵗ) if k != j_index; init=1.0)
                end
            end
            Bᵗ[:, :, xᵢᵗ, :, :] *= ϕᵢ[t][xᵢᵗ]
        end
        B[t] = Bᵗ
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        next!(prog, showvalues=[(:t, "$t/$T")])
    end

    return MPEM3(B), 0.0
end

# compute outgoing message to dummy neighbor to get the belief
function f_bp_dummy_neighbor(A::Vector{MPEM2{F}}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0)) where {F,U<:BPFactor}
    T = getT(A[1])
    @assert all(getT(a) == T for a in A)
    q = length(ϕᵢ[1])
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1
    z = length(A)      # z = |∂i|
    xₙᵢ = Iterators.product((1:size(ψₙᵢ[k][1],2) for k=1:z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T + 1)
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    for t in 1:T+1
        # select incoming A's but not the j-th one
        Aᵗ = kron2([A[k][t] for k in eachindex(A)]...)
        nrows = size(Aᵗ, 1)
        ncols = size(Aᵗ, 2)
        Bᵗ = zeros(nrows, ncols, q, 1, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xₙᵢᵗ in xₙᵢ
                    Bᵗ[:, :, xᵢᵗ, 1, xᵢᵗ⁺¹] .+= (t == T + 1 ? 1.0 : wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) .*
                                                    Aᵗ[:, :, xᵢᵗ, xₙᵢᵗ...] .* ϕᵢ[t][xᵢᵗ] .*
                                                    prod(ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in enumerate(xₙᵢᵗ))
                end
            end
        end
        B[t] = Bᵗ
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        next!(prog, showvalues=[(:t, "$t/$T")])
    end

    return MPEM3(B), 0.0
end

function accumulate_L(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)
    T = getT(Aᵢⱼ)
    @assert getT(Aⱼᵢ) == T
    L = [zeros(0, 0) for _ in 0:T]
    Aᵢⱼ⁰ = Aᵢⱼ[begin]; Aⱼᵢ⁰ = Aⱼᵢ[begin]; ψᵢⱼ⁰ = ψᵢⱼ[begin] 
    @tullio L⁰[a¹, b¹] := Aᵢⱼ⁰[1, a¹, xᵢ⁰, xⱼ⁰] * ψᵢⱼ⁰[xᵢ⁰, xⱼ⁰] * Aⱼᵢ⁰[1, b¹, xⱼ⁰, xᵢ⁰]
    L[1] = L⁰

    for t in 1:T
        Aᵢⱼᵗ = Aᵢⱼ[t+1]; Aⱼᵢᵗ = Aⱼᵢ[t+1]; ψᵢⱼᵗ = ψᵢⱼ[t+1]; Lᵗ = L[t]
        @tullio Lᵗ⁺¹[aᵗ⁺¹, bᵗ⁺¹] := Lᵗ[aᵗ, bᵗ] * Aᵢⱼᵗ[aᵗ, aᵗ⁺¹, xᵢᵗ, xⱼᵗ] * ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ] * Aⱼᵢᵗ[bᵗ, bᵗ⁺¹, xⱼᵗ, xᵢᵗ]
        L[t+1] = Lᵗ⁺¹
    end
    return L
end

function accumulate_R(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)
    T = getT(Aᵢⱼ)
    @assert getT(Aⱼᵢ) == T
    R = [zeros(0, 0) for _ in 0:T]
    Aᵢⱼᵀ = Aᵢⱼ[end]; Aⱼᵢᵀ = Aⱼᵢ[end]; ψᵢⱼᵀ = ψᵢⱼ[end]
    @tullio Rᵀ[aᵀ, bᵀ] := Aᵢⱼᵀ[aᵀ, 1, xᵢᵀ, xⱼᵀ] * ψᵢⱼᵀ[xᵢᵀ, xⱼᵀ] * Aⱼᵢᵀ[bᵀ, 1, xⱼᵀ, xᵢᵀ]
    R[end] = Rᵀ

    for t in T:-1:1
        Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]; ψᵢⱼᵗ = ψᵢⱼ[t]; Rᵗ⁺¹ = R[t+1]
        @tullio Rᵗ[aᵗ, bᵗ] := Aᵢⱼᵗ[aᵗ, aᵗ⁺¹, xᵢᵗ, xⱼᵗ] * Aⱼᵢᵗ[bᵗ, bᵗ⁺¹, xⱼᵗ, xᵢᵗ] * ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ] * Rᵗ⁺¹[aᵗ⁺¹, bᵗ⁺¹]
        R[t] = Rᵗ
    end
    return R
end

function accumulate_M(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)
    T = getT(Aᵢⱼ)
    @assert getT(Aⱼᵢ) == T
    M = [zeros(0, 0, 0, 0) for _ in 0:T, _ in 0:T]

    # initial condition
    for t in 1:T
        range_aᵗ⁺¹ = axes(Aᵢⱼ[t+1], 1)
        range_bᵗ⁺¹ = axes(Aⱼᵢ[t+1], 1)
        Mᵗᵗ⁺¹ = [float((a == c) * (b == d)) for a in range_aᵗ⁺¹, c in range_aᵗ⁺¹, b in range_bᵗ⁺¹, d in range_bᵗ⁺¹]
        M[t, t+1] = Mᵗᵗ⁺¹
    end

    for t in 1:T
        Mᵗᵘ⁻¹ = M[t, t+1]
        for u in t+2:T+1
            Aᵢⱼᵘ⁻¹ = Aᵢⱼ[u-1]; Aⱼᵢᵘ⁻¹ = Aⱼᵢ[u-1]; ψᵢⱼᵘ⁻¹ = ψᵢⱼ[u-1]
            @reduce Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ, bᵗ⁺¹, bᵘ] |= sum(aᵘ⁻¹, bᵘ⁻¹, xᵢᵘ⁻¹, xⱼᵘ⁻¹) Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹, bᵗ⁺¹, bᵘ⁻¹] * 
                Aᵢⱼᵘ⁻¹[aᵘ⁻¹, aᵘ, xᵢᵘ⁻¹, xⱼᵘ⁻¹] * ψᵢⱼᵘ⁻¹[xᵢᵘ⁻¹, xⱼᵘ⁻¹] * Aⱼᵢᵘ⁻¹[bᵘ⁻¹, bᵘ, xⱼᵘ⁻¹, xᵢᵘ⁻¹]
            M[t, u] = Mᵗᵘ⁻¹
        end
    end

    return M
end


function pair_belief_tu(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ; showprogress::Bool=false)

    L = accumulate_L(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    R = accumulate_R(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    M = accumulate_M(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)

    T = getT(Aᵢⱼ)
    @assert getT(Aⱼᵢ) == T
    q = size(Aᵢⱼ[1], 3)
    b = [zeros(q, q, q, q) for _ in 0:T, _ in 0:T]

    dt = showprogress ? 0.1 : Inf
    prog = Progress(Int(T * (T + 1) / 2); dt, desc="Computing beliefs at pairs of times (t,u)")
    for t in 1:T
        Lᵗ⁻¹ = t == 1 ? [1.0;;] : L[t-1]
        Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]; ψᵢⱼᵗ = ψᵢⱼ[t]
        for u in t+1:T+1
            Rᵘ⁺¹ = u == T + 1 ? [1.0;;] : R[u+1]
            Aᵢⱼᵘ = Aᵢⱼ[u]; Aⱼᵢᵘ = Aⱼᵢ[u]; ψᵢⱼᵘ = ψᵢⱼ[u]
            Mᵗᵘ = M[t, u]
            @tullio bᵗᵘ[xᵢᵗ, xⱼᵗ, xᵢᵘ, xⱼᵘ] :=
                Lᵗ⁻¹[aᵗ, bᵗ] * Aᵢⱼᵗ[aᵗ, aᵗ⁺¹, xᵢᵗ, xⱼᵗ] * ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ] *
                Aⱼᵢᵗ[bᵗ, bᵗ⁺¹, xⱼᵗ, xᵢᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ, bᵗ⁺¹, bᵘ] * Aᵢⱼᵘ[aᵘ, aᵘ⁺¹, xᵢᵘ, xⱼᵘ] *
                ψᵢⱼᵘ[xᵢᵘ, xⱼᵘ] * Aⱼᵢᵘ[bᵘ, bᵘ⁺¹, xⱼᵘ, xᵢᵘ] * Rᵘ⁺¹[aᵘ⁺¹, bᵘ⁺¹]
            b[t, u] .= bᵗᵘ ./ sum(bᵗᵘ)
            next!(prog, showvalues=[(:t, t), (:u, u)])
        end
    end

    return b
end

# compute bᵢⱼᵗ(xᵢᵗ,xⱼᵗ) from μᵢⱼ and μⱼᵢ
# also return normalization zᵢⱼ
function pair_belief(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)

    L = accumulate_L(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    R = accumulate_R(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    z = only(L[end])
    @assert only(R[begin]) ≈ z
    z ≥ 0 || @warn "z=$z"

    T = getT(Aᵢⱼ)
    @assert getT(Aⱼᵢ) == T

    Aᵢⱼ⁰ = Aᵢⱼ[begin]; Aⱼᵢ⁰ = Aⱼᵢ[begin]; ψᵢⱼ⁰ = ψᵢⱼ[begin]
    R¹ = R[2]
    @reduce b⁰[xᵢ⁰, xⱼ⁰] := sum(a¹, b¹) Aᵢⱼ⁰[1, a¹, xᵢ⁰, xⱼ⁰] * ψᵢⱼ⁰[xᵢ⁰, xⱼ⁰] *
        Aⱼᵢ⁰[1, b¹, xⱼ⁰, xᵢ⁰] * R¹[a¹, b¹]
    b⁰ ./= sum(b⁰)

    Aᵢⱼᵀ = Aᵢⱼ[end]; Aⱼᵢᵀ = Aⱼᵢ[end]; ψᵢⱼᵀ = ψᵢⱼ[end]
    Lᵀ⁻¹ = L[end-1]
    @reduce bᵀ[xᵢᵀ, xⱼᵀ] := sum(aᵀ, bᵀ) Lᵀ⁻¹[aᵀ, bᵀ] * Aᵢⱼᵀ[aᵀ, 1, xᵢᵀ, xⱼᵀ] *
        ψᵢⱼᵀ[xᵢᵀ, xⱼᵀ] * Aⱼᵢᵀ[bᵀ, 1, xⱼᵀ, xᵢᵀ]
    bᵀ ./= sum(bᵀ)

    b = map(2:T) do t
        Lᵗ⁻¹ = L[t-1]
        Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]; ψᵢⱼᵗ = ψᵢⱼ[t]
        Rᵗ⁺¹ = R[t+1]
        @reduce bᵗ[xᵢᵗ, xⱼᵗ] := sum(aᵗ, aᵗ⁺¹, bᵗ, bᵗ⁺¹) Lᵗ⁻¹[aᵗ, bᵗ] *
                                Aᵢⱼᵗ[aᵗ, aᵗ⁺¹, xᵢᵗ, xⱼᵗ] * Aⱼᵢᵗ[bᵗ, bᵗ⁺¹, xⱼᵗ, xᵢᵗ] * 
                                ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ] * Rᵗ⁺¹[aᵗ⁺¹, bᵗ⁺¹]
        bᵗ ./= sum(bᵗ)
    end

    return [b⁰, b..., bᵀ], z
end