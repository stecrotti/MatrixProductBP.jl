# for a `SimpleBPFactor`, outgoing messages can be computed recursively
abstract type SimpleBPFactor <: BPFactor; end

function prob_ijy(::Type{<:SimpleBPFactor})
    error("Not implemented")
end

function prob_ijy_dummy(::Type{<:SimpleBPFactor})
    error("Not implemented")
end

function f_bp(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, 
        wᵢ::Vector{U}, ϕᵢ, ψₙᵢ, j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F,U<:SimpleBPFactor}

    d = length(A) - 1   # number of neighbors other than j
    @assert j ∈ eachindex(A)

    # initialize recursion
    M = reshape(vcat(ones(1,q), zeros(q-1,q)), (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    l = 1
    for k in eachindex(A)
        k == j && continue
        mᵢⱼₗ₁ = f_bp_partial(A[k], mᵢⱼₗ₁, wᵢ, l)
        l += 1
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, pᵢ⁰, wᵢ, ϕᵢ, d; prob = prob_ijy(U))

    return B
end


function f_bp_dummy_neighbor(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, 
        wᵢ::Vector{U}, ϕᵢ, ψₙᵢ;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F,U<:SimpleBPFactor}
    
    d = length(A)

    # initialize recursion
    M = reshape(vcat(ones(1,q), zeros(q-1,q)), (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    # compute partial messages from all neighbors
    for l in eachindex(A)
        mᵢⱼₗ₁ = f_bp_partial(A[l], mᵢⱼₗ₁, U, l)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, pᵢ⁰, wᵢ, ϕᵢ, d; prob = prob_ijy_dummy(U))

    return B
end

function beliefs(bp::MPBP{q,T,F,<:SimpleBPFactor};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    b = [[zeros(q) for _ in 0:T] for _ in vertices(bp.g)]
    for i in eachindex(b)
        A = onebpiter_dummy_neighbor(bp, i; svd_trunc)
        b[i] .= firstvar_marginal(A)
    end
    b
end

function beliefs_tu(bp::MPBP{q,T,F,<:SimpleBPFactor};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    b = [[zeros(q, q) for _ in 0:T, _ in 0:T] for _ in vertices(bp.g)]
    for i in eachindex(b)
        A = onebpiter_dummy_neighbor(bp, i; svd_trunc)
        b[i] .= firstvar_marginal_tu(A)
    end
    b
end