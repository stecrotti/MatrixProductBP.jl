# compute outgoing message as a function of the incoming ones
# A is a vector with all incoming messages. At index j_index there is m(j → i)
# ψᵢⱼ are the ones living on the outedges of node i
function f_bp_generic(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j_index::Integer;
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0)) where {q,T,F}
    @assert length(pᵢ⁰) == q
    @assert length(wᵢ) == T
    @assert length(ϕᵢ) == T + 1 
    @assert j_index in eachindex(A)
    z = length(A)      # z = |∂i|
    x_neigs = Iterators.product(fill(1:q, z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T+1)
    A⁰ = kron2([A[k][begin] for k in eachindex(A)[Not(j_index)]]...)
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)
    
    for xᵢ⁰ in 1:q
        for xᵢ¹ in 1:q
            for xₙᵢ⁰ in x_neigs
                xⱼ⁰ = xₙᵢ⁰[j_index]
                xₙᵢ₋ⱼ⁰ = xₙᵢ⁰[Not(j_index)]
                for a¹ in axes(A⁰, 2)
                    B⁰[xᵢ⁰,xⱼ⁰,1,a¹,xᵢ¹] += wᵢ[1](xᵢ¹, xₙᵢ⁰,xᵢ⁰) *
                        A⁰[1,a¹,xᵢ⁰,xₙᵢ₋ⱼ⁰...] * 
                        prod(sqrt, ψₙᵢ[k][begin][xᵢ⁰,xₖ⁰] for (k,xₖ⁰) in enumerate(xₙᵢ⁰))
                end
            end
        end
        B⁰[xᵢ⁰,:,:,:,:] .*= pᵢ⁰[xᵢ⁰]  * ϕᵢ[begin][xᵢ⁰]
    end
    B[begin] = B⁰

    dt = showprogress ? 1.0 : Inf
    prog = Progress(T-1, dt=dt, desc="Computing outgoing message")
    for t in 1:T-1
        # select incoming A's but not the j-th one
        Aᵗ = kron2([A[k][begin+t] for k in eachindex(A)[Not(j_index)]]...)
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xₙᵢᵗ in x_neigs
                    xⱼᵗ = xₙᵢᵗ[j_index]
                    xₙᵢ₋ⱼᵗ = xₙᵢᵗ[Not(j_index)]
                    Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= wᵢ[t+1](xᵢᵗ⁺¹,xₙᵢᵗ,xᵢᵗ) *
                            Aᵗ[:,:,xᵢᵗ,xₙᵢ₋ⱼᵗ...] *
                            prod(sqrt, ψₙᵢ[k][t+1][xᵢᵗ,xₖᵗ] for (k,xₖᵗ) in enumerate(xₙᵢᵗ))
                end
            end
            Bᵗ[xᵢᵗ,:,:,:,:] *= ϕᵢ[t+1][xᵢᵗ]
        end
        B[begin+t] = Bᵗ
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        next!(prog, showvalues=[(:t,"$t/$T")])
    end

    # select incoming A's but not the j-th one
    Aᵀ = kron2([A[k][begin+T] for k in eachindex(A)[Not(j_index)]]...)
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)

    for xᵢᵀ in 1:q
        for xᵢᵀ⁺¹ in 1:q
            for xₙᵢᵀ in x_neigs
                xⱼᵀ = xₙᵢᵀ[j_index]
                xₙᵢ₋ⱼᵀ = xₙᵢᵀ[Not(j_index)]
                Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= 
                        Aᵀ[:,:,xᵢᵀ,xₙᵢ₋ⱼᵀ...] *
                        prod(sqrt, ψₙᵢ[k][end][xᵢᵀ,xₖᵀ] for (k,xₖᵀ) in enumerate(xₙᵢᵀ))
            end
        end
        Bᵀ[xᵢᵀ,:,:,:,:] *= ϕᵢ[end][xᵢᵀ]
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

    return MPEM3(B)
end

function accumulate_L(Aᵢⱼ::MPEM2{q,T,F}, Aⱼᵢ::MPEM2{q,T,F}) where {q,T,F}
    L = [zeros(0,0) for t in 0:T]
    Aᵢⱼ⁰ = Aᵢⱼ[begin]; Aⱼᵢ⁰ = Aⱼᵢ[begin]
    @tullio L⁰[a¹, b¹] := Aᵢⱼ⁰[1,a¹,xᵢ⁰,xⱼ⁰] * Aⱼᵢ⁰[1,b¹,xⱼ⁰,xᵢ⁰]
    L[1] = L⁰

    Lᵗ = L⁰
    for t in 1:T
        Aᵢⱼᵗ = Aᵢⱼ[t+1]; Aⱼᵢᵗ = Aⱼᵢ[t+1]
        @reduce Lᵗ[aᵗ⁺¹,bᵗ⁺¹] |= sum(xᵢᵗ,xⱼᵗ,aᵗ,bᵗ)  Lᵗ[aᵗ,bᵗ] * Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ]
        L[t+1] = Lᵗ
    end
    return L
end

function accumulate_R(Aᵢⱼ::MPEM2{q,T,F}, Aⱼᵢ::MPEM2{q,T,F}) where {q,T,F}
    R = [zeros(0,0) for t in 0:T]
    Aᵢⱼᵀ = Aᵢⱼ[end]; Aⱼᵢᵀ = Aⱼᵢ[end]
    @tullio Rᵀ[aᵀ, bᵀ] := Aᵢⱼᵀ[aᵀ,1,xᵢᵀ,xⱼᵀ] * Aⱼᵢᵀ[bᵀ,1,xⱼᵀ,xᵢᵀ]
    R[end] = Rᵀ

    Rᵗ = Rᵀ
    for t in T:-1:1
        Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]
        @reduce Rᵗ[aᵗ,bᵗ] |= sum(xᵢᵗ,xⱼᵗ,aᵗ⁺¹,bᵗ⁺¹)  Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * Rᵗ[aᵗ⁺¹,bᵗ⁺¹]
        R[t] = Rᵗ
    end
    return R
end

function accumulate_M(Aᵢⱼ::MPEM2{q,T,F}, Aⱼᵢ::MPEM2{q,T,F}) where {q,T,F}
    M = [zeros(0,0,0,0) for _ in 0:T, _ in 0:T]

    # initial condition
    for t in 1:T
        range_aᵗ⁺¹ = axes(Aᵢⱼ[t+1], 1)
        range_bᵗ⁺¹ = axes(Aⱼᵢ[t+1], 1)
        Mᵗᵗ⁺¹ = [float((a==c)*(b==d)) for a in range_aᵗ⁺¹, c in range_aᵗ⁺¹, b in range_bᵗ⁺¹, d in range_bᵗ⁺¹]
        M[t,t+1] = Mᵗᵗ⁺¹
    end

    for t in 1:T
        Mᵗᵘ⁻¹ = M[t,t+1]
        for u in t+2:T+1
            Aᵢⱼᵘ⁻¹ = Aᵢⱼ[u-1]; Aⱼᵢᵘ⁻¹ = Aⱼᵢ[u-1]
            @reduce Mᵗᵘ⁻¹[aᵗ⁺¹,aᵘ,bᵗ⁺¹,bᵘ] |= sum(aᵘ⁻¹,bᵘ⁻¹,xᵢᵘ⁻¹,xⱼᵘ⁻¹) Mᵗᵘ⁻¹[aᵗ⁺¹,aᵘ⁻¹,bᵗ⁺¹,bᵘ⁻¹] * Aᵢⱼᵘ⁻¹[aᵘ⁻¹,aᵘ,xᵢᵘ⁻¹,xⱼᵘ⁻¹] * Aⱼᵢᵘ⁻¹[bᵘ⁻¹,bᵘ,xⱼᵘ⁻¹,xᵢᵘ⁻¹]
            M[t,u] = Mᵗᵘ⁻¹
        end
    end

    return M
end


function pair_belief_tu(Aᵢⱼ::MPEM2{q,T,F}, Aⱼᵢ::MPEM2{q,T,F};
        showprogress::Bool=false) where {q,T,F}
    
    L = accumulate_L(Aᵢⱼ, Aⱼᵢ); R = accumulate_R(Aᵢⱼ, Aⱼᵢ)
    M = accumulate_M(Aᵢⱼ, Aⱼᵢ)    
    b = [zeros(q,q,q,q) for _ in 0:T, _ in 0:T]

    dt = showprogress ? 0.1 : Inf
    prog = Progress(Int(T*(T+1)/2); dt, desc="Computing beliefs at pairs of times (t,u)") 
    for t in 1:T
        Lᵗ⁻¹ = t==1 ? [1.0;;] : L[t-1]; 
        Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]
        for u in t+1:T+1
            Rᵘ⁺¹ = u==T+1 ? [1.0;;] : R[u+1]; 
            Aᵢⱼᵘ = Aᵢⱼ[u]; Aⱼᵢᵘ = Aⱼᵢ[u]
            Mᵗᵘ = M[t,u]
            @tullio bᵗᵘ[xᵢᵗ, xⱼᵗ, xᵢᵘ, xⱼᵘ] := 
                Lᵗ⁻¹[aᵗ,bᵗ] * Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] *
                Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * Mᵗᵘ[aᵗ⁺¹,aᵘ,bᵗ⁺¹,bᵘ] * Aᵢⱼᵘ[aᵘ,aᵘ⁺¹,xᵢᵘ,xⱼᵘ] *
                Aⱼᵢᵘ[bᵘ,bᵘ⁺¹,xⱼᵘ,xᵢᵘ] * Rᵘ⁺¹[aᵘ⁺¹,bᵘ⁺¹]
            b[t,u] .= bᵗᵘ ./ sum(bᵗᵘ)
            next!(prog, showvalues=[(:t,t), (:u,u)])
        end
    end

    return b
end

# compute bᵢⱼᵗ(xᵢᵗ,xⱼᵗ) from μᵢⱼ and μⱼᵢ
# also return normalization zᵢⱼ
function pair_belief(Aᵢⱼ::MPEM2{q,T,F}, Aⱼᵢ::MPEM2{q,T,F}) where {q,T,F}

    L = accumulate_L(Aᵢⱼ, Aⱼᵢ); R = accumulate_R(Aᵢⱼ, Aⱼᵢ)
    z = only(L[end])
    @assert only(R[begin]) ≈ z
    z ≥ 0 || @warn "z=$z"

    Aᵢⱼ⁰ = Aᵢⱼ[begin]; Aⱼᵢ⁰ = Aⱼᵢ[begin]; R¹ = R[2]
    @reduce b⁰[xᵢ⁰,xⱼ⁰] := sum(a¹,b¹) Aᵢⱼ⁰[1,a¹,xᵢ⁰,xⱼ⁰] * Aⱼᵢ⁰[1,b¹,xⱼ⁰,xᵢ⁰] * R¹[a¹,b¹]
    b⁰ ./= sum(b⁰)

    Aᵢⱼᵀ = Aᵢⱼ[end]; Aⱼᵢᵀ = Aⱼᵢ[end]; Lᵀ⁻¹ = L[end-1]
    @reduce bᵀ[xᵢᵀ,xⱼᵀ] := sum(aᵀ,bᵀ) Lᵀ⁻¹[aᵀ,bᵀ] * Aᵢⱼᵀ[aᵀ,1,xᵢᵀ,xⱼᵀ] * Aⱼᵢᵀ[bᵀ,1,xⱼᵀ,xᵢᵀ]
    bᵀ ./= sum(bᵀ)

    b = map(2:T) do t 
        Lᵗ⁻¹ = L[t-1]; Aᵢⱼᵗ = Aᵢⱼ[t]; Aⱼᵢᵗ = Aⱼᵢ[t]; Rᵗ⁺¹ = R[t+1]
        @reduce bᵗ[xᵢᵗ,xⱼᵗ] := sum(aᵗ,aᵗ⁺¹,bᵗ,bᵗ⁺¹) Lᵗ⁻¹[aᵗ,bᵗ] * 
            Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xⱼᵗ] * Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * Rᵗ⁺¹[aᵗ⁺¹,bᵗ⁺¹]  
        bᵗ ./= sum(bᵗ)
    end

    return [b⁰, b..., bᵀ], z
end



# #### OLD
# # compute belief as an outgoing message to a dummy neigbor
# # A is a vector with all incoming messages.
# function f_bp(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ;
#         showprogress=false) where {q,T,F}
#     @assert length(pᵢ⁰) == q
#     @assert length(wᵢ) == T
#     @assert length(ϕᵢ) == T
#     z = length(A)      # z = |∂i|
#     x_neigs = Iterators.product(fill(1:q, z)...) .|> collect

#     B = Vector{Array{F,5}}(undef, T+1)
#     A⁰ = kron2([A[k][begin] for k in eachindex(A)]...)
#     nrows = size(A⁰, 1); ncols = size(A⁰, 2)
#     B⁰ = zeros(q, q, nrows, ncols, q)

#     for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
#         for xₙᵢ⁰ in x_neigs
#             for xⱼ⁰ in 1:q
#                 for a¹ in axes(A⁰, 2)
#                     B⁰[xᵢ⁰,xⱼ⁰,1,a¹,xᵢ¹] += wᵢ[1](xᵢ¹, xₙᵢ⁰,xᵢ⁰) *
#                         A⁰[1,a¹,xᵢ⁰,xₙᵢ⁰...]
#                 end
#             end
#         end
#         B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
#     end
#     B[1] = B⁰

#     dt = showprogress ? 1.0 : Inf
#     prog = Progress(T-1, dt=dt, desc="Computing belief")
#     for t in 1:T-1
#         # select incoming A's but not the j-th one
#         Aᵗ = kron2([A[k][begin+t] for k in eachindex(A)]...)
#         nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
#         Bᵗ = zeros(q, q, nrows, ncols, q)

#         for xᵢᵗ⁺¹ in 1:q
#             for xᵢᵗ in 1:q
#                 # for xⱼᵗ in 1:q
#                     for xₙᵢᵗ in x_neigs
#                         xⱼᵗ = xₙᵢᵗ[1]
#                         # for aᵗ in axes(Aᵗ, 1), aᵗ⁺¹ in axes(Aᵗ, 2)
#                         #     Bᵗ[xᵢᵗ,xⱼᵗ,aᵗ,aᵗ⁺¹,xᵢᵗ⁺¹] += wᵢ[t+1](xᵢᵗ⁺¹,xₙᵢᵗ,xᵢᵗ) *
#                         #         Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xₙᵢᵗ...]
#                         # end
#                         Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= wᵢ[t+1](xᵢᵗ⁺¹,xₙᵢᵗ,xᵢᵗ) *
#                                 Aᵗ[:,:,xᵢᵗ,xₙᵢᵗ...]
#                     end
#                 # end
#             end
#             Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+1][xᵢᵗ⁺¹]
#         end
#         B[t+1] = Bᵗ
#         next!(prog, showvalues=[(:t,"$t/$T")])
#     end

#     Aᵀ = kron2([A[k][end] for k in eachindex(A)]...)
#     nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
#     Bᵀ = ones(q, q, nrows, ncols, q)


#     if z != 0
#         Aᵀ_reshaped = reshape(Aᵀ, size(Aᵀ)[1:3]..., prod(size(Aᵀ)[4:end]))
#         Aᵀ_reshaped_summed = sum(Aᵀ_reshaped, dims=4)[:,1,:,1]
#         # Bᵀ has some redundant dimensions, like the 5-th. We decided that it should 
#         #  take the same values no matter the 5-th index
#         # Same goes for the 2nd index, i.e. xⱼᵀ, on which Bᵀ does not depend.
#         # Transpose to match indices aᵀ and xᵢᵀ.
#         for xⱼᵀ in 1:q, xᵢᵀ⁺¹ in 1:q
#             Bᵀ[:,xⱼᵀ,:,1,xᵢᵀ⁺¹] .= Aᵀ_reshaped_summed'
#         end
#     end
#     B[end] = Bᵀ
#     return MPEM3(B)
# end